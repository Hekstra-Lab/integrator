import argparse
import gc
import os
import traceback

import numpy as np
import torch
from dials.array_family import flex
from dials.util import Sorry
from dials.util.options import ArgumentParser, flatten_experiments, flatten_reflections
from libtbx.phil import parse

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def to_gpu(data):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        return data


phil_scope = parse(
    """
  nx = 1
    .type = int(value_min=1)
    .help = "+/- x around centroid"

  ny = 1
    .type = int(value_min=1)
    .help = "+/- y around centroid"

  nz = 1
    .type = int(value_min=1)
    .help = "+/- z around centroid"

  output {
    reflections = 'shoeboxes.refl'
      .type = str
      .help = "The integrated output filename"
  }
"""
)

# Create the parser
parser = ArgumentParser(
    phil=phil_scope,
    read_experiments=True,
    read_reflections=True,
)

# arguments and options
params, options = parser.parse_args(
    ["integrated.refl", "integrated.expt", "nx=10", "ny=10", "nz=1"]
)

# reflections
reflections = flatten_reflections(params.input.reflections)

# experiments
experiments = flatten_experiments(params.input.experiments)

diff_phil = parser.diff_phil.as_str()

if not any([experiments, reflections]):
    parser.print_help()
    exit(0)
elif len(experiments) > 1:
    raise Sorry("More than 1 experiment set")
elif len(experiments) == 1:
    imageset = experiments[0].imageset
if len(reflections) != 1:
    raise Sorry("Need 1 reflection table, got %d" % len(reflections))
else:
    reflections = reflections[0]

# Check the reflections contain the necessary stuff
assert "bbox" in reflections
assert "panel" in reflections

# Get some models
detector = imageset.get_detector()
scan = imageset.get_scan()
frame0, frame1 = scan.get_array_range()

x, y, z = reflections["xyzcal.px"].parts()

x = flex.floor(x).iround()
y = flex.floor(y).iround()
z = flex.floor(z).iround()

bbox = flex.int6(len(reflections))

dx, dy = detector[0].get_image_size()

for j, (_x, _y, _z) in enumerate(zip(x, y, z, strict=False)):
    x0 = _x - params.nx
    x1 = _x + params.nx + 1
    y0 = _y - params.ny
    y1 = _y + params.ny + 1
    z0 = _z - params.nz
    z1 = _z + params.nz + 1
    if x0 < 0:
        x0 = 0
    if x1 >= dx:
        x1 = dx - 1
    if y0 < 0:
        y0 = 0
    if y1 >= dy:
        y1 = dy - 1
    if z0 < frame0:
        z0 = frame0
    if z1 >= frame1:
        z1 = frame1 - 1
    bbox[j] = (x0, x1, y0, y1, z0, z1)

reflections["bbox"] = bbox

# beware change of variables - removing those which were a different shape because boundary conditions
x0, x1, y0, y1, z0, z1 = bbox.parts()

dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)

good = (dx == 2 * params.nx + 1) & (dy == 2 * params.ny + 1) & (dz == 2 * params.nz + 1)

print(f"{good.count(True)} / {good.size()} kept")

reflections = reflections.select(good)

# store shoeboxes into reflection file
reflections["shoebox"] = flex.shoebox(reflections["panel"], reflections["bbox"], allocate=True)

# get shoeboxes
reflections.extract_shoeboxes(imageset)

# assign an identifier to each reflection
reflections["refl_ids"] = flex.int(np.arange(len(reflections)))

for experiment, indices in reflections.iterate_experiments_and_indices(experiments):
    print(experiment)

# get refls by index
subset = reflections.select(indices)

# mask the neighbouring reflections
modified_count = 0

for experiment, indices in reflections.iterate_experiments_and_indices(experiments):
    subset = reflections.select(indices)
    modified = subset["shoebox"].mask_neighbouring(
        subset["miller_index"],
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.scan,
        experiment.crystal,
    )
    modified_count += modified.count(True)


def process_and_save_chunk(chunk, chunk_index, output_dir, overwrite=False):
    masks_file = os.path.join(output_dir, f"masks_{chunk_index}.pt")
    samples_file = os.path.join(output_dir, f"samples_{chunk_index}.pt")
    metadata_file = os.path.join(output_dir, f"metadata_{chunk_index}.pt")

    if not overwrite and all(os.path.exists(f) for f in [masks_file, samples_file, metadata_file]):
        print(f"Chunk {chunk_index} already processed. Skipping.")
        return 0

    batch_size = 1000  # Process in very small batches
    total_processed = 0

    all_filtered_masks = []
    all_filtered_samples = []
    all_filtered_metadata = []

    for i in range(0, len(chunk), batch_size):
        sub_chunk = chunk[i : i + batch_size]

        try:
            print(f"Processing sub-chunk {i // batch_size + 1} of chunk {chunk_index}")

            print("Creating masks...")
            masks = torch.stack(
                [
                    torch.tensor(sbox.mask.as_numpy_array().ravel(), dtype=torch.float32)
                    for sbox in sub_chunk["shoebox"]
                ]
            )
            masks[masks == 3] = 0

            print("Creating coordinates...")
            coordinates = torch.stack(
                [
                    torch.tensor(sbox.coords().as_numpy_array(), dtype=torch.float32)
                    for sbox in sub_chunk["shoebox"]
                ]
            )

            print("Creating i_obs...")
            i_obs = torch.stack(
                [
                    torch.tensor(
                        sbox.data.as_numpy_array().ravel().astype(np.float32), dtype=torch.float32
                    )
                    for sbox in sub_chunk["shoebox"]
                ]
            ).unsqueeze(-1)

            print("Creating centroids...")
            centroids = torch.tensor(
                sub_chunk["xyzobs.px.value"].as_numpy_array(), dtype=torch.float32
            ).unsqueeze(1)

            print("Calculating dxyz...")
            try:
                coordinates = to_gpu(coordinates)
                centroids = to_gpu(centroids)
                dxyz = torch.abs(coordinates - centroids)
                dxyz = dxyz.cpu()
            except RuntimeError as e:
                print(f"GPU processing failed: {e}")
                print("Falling back to CPU processing")
                dxyz = torch.abs(coordinates - centroids)

            coordinates = coordinates.cpu()
            centroids = centroids.cpu()
            torch.cuda.empty_cache()

            print("Concatenating samples...")
            samples = torch.cat((coordinates, dxyz, i_obs), dim=-1)

            del coordinates, dxyz, i_obs, centroids
            gc.collect()

            print("Filtering dead panels...")
            filter = (samples[..., -1] < 0).sum(-1) < 700

            filtered_samples = torch.clamp(samples[filter], min=0)
            filtered_masks = masks[filter]

            del samples, masks
            gc.collect()

            print("Creating metadata...")
            metadata = torch.stack(
                [
                    torch.tensor(sub_chunk[col].as_numpy_array(), dtype=torch.float32)
                    for col in [
                        "intensity.sum.value",
                        "intensity.sum.variance",
                        "intensity.prf.value",
                        "intensity.prf.variance",
                        "refl_ids",
                    ]
                ]
            ).transpose(0, 1)

            filtered_metadata = metadata[filter]

            del metadata
            gc.collect()

            print("Appending to lists...")
            all_filtered_masks.append(filtered_masks)
            all_filtered_samples.append(filtered_samples)
            all_filtered_metadata.append(filtered_metadata)

            total_processed += filtered_samples.shape[0]

            del filtered_samples, filtered_masks, filtered_metadata
            gc.collect()
            torch.cuda.empty_cache()

            print(f"Sub-chunk {i // batch_size + 1} of chunk {chunk_index} processed successfully")

        except Exception as e:
            print(f"Error processing sub-chunk {i // batch_size + 1} of chunk {chunk_index}")
            print(f"Error details: {str(e)}")
            print(traceback.format_exc())
            continue

    if not all_filtered_masks:
        print(f"No data processed for chunk {chunk_index}")
        return 0, None, None, None

    try:
        print(f"Combining data for chunk {chunk_index}...")
        combined_masks = torch.cat(all_filtered_masks, dim=0)
        combined_samples = torch.cat(all_filtered_samples, dim=0)
        combined_metadata = torch.cat(all_filtered_metadata, dim=0)

        torch.save(combined_masks, masks_file)
        torch.save(combined_samples, samples_file)
        torch.save(combined_metadata, metadata_file)

        del all_filtered_masks, all_filtered_samples, all_filtered_metadata
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Chunk {chunk_index} processed and saved successfully")
    except Exception as e:
        print(f"Error saving data for chunk {chunk_index}")
        print(f"Error details: {str(e)}")
        print(traceback.format_exc())
        return 0, None, None, None

    return total_processed, combined_masks, combined_samples, combined_metadata


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process reflection data")
parser.add_argument(
    "--output_dir", type=str, default="processed_data", help="Output directory for processed files"
)
args = parser.parse_args()

# Main processing loop
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

chunk_size = 50000  # Larger chunk size, but processed in smaller batches
total_processed = 0
chunk_index = 0

all_masks = []
all_samples = []
all_metadata = []

processed_indices = []

for i in range(0, len(reflections), chunk_size):
    chunk = reflections[i : i + chunk_size]
    processed_count, masks, samples, metadata = process_chunk(chunk, chunk_index, output_dir)

    if masks is not None:
        all_masks.append(masks)
        all_samples.append(samples)
        all_metadata.append(metadata)
        processed_indices.extend(range(i, i + processed_count))

    total_processed += processed_count
    chunk_index += 1
    print(f"Processed chunk {chunk_index}, total processed: {total_processed}")

    gc.collect()
    torch.cuda.empty_cache()

# Combine all processed data
print("Combining all processed data...")
combined_masks = torch.cat(all_masks, dim=0)
combined_samples = torch.cat(all_samples, dim=0)
combined_metadata = torch.cat(all_metadata, dim=0)

# Save combined data
torch.save(combined_masks, os.path.join(output_dir, "combined_masks.pt"))
torch.save(combined_samples, os.path.join(output_dir, "combined_samples.pt"))
torch.save(combined_metadata, os.path.join(output_dir, "combined_metadata.pt"))

# Create and save the final reflection file
final_reflections = reflections.select(flex.size_t(processed_indices))

# Remove the shoebox column to save memory
if "shoebox" in final_reflections:
    del final_reflections["shoebox"]

final_reflections.as_file(os.path.join(output_dir, "reflections_.refl"))

print(f"Processing completed. Total samples processed: {total_processed}")
print(f"Final reflection file saved: {os.path.join(output_dir, 'reflections_.refl')}")
