import gc

import numpy as np
import torch
from dials.array_family import flex
from dials.util.options import ArgumentParser, flatten_experiments, flatten_reflections
from scipy.spatial import KDTree

parser = ArgumentParser(read_experiments=True, read_reflections=True)
params, options = parser.parse_args(["./poly_refined_HEWL_anom_3049.expt", "./integrated.refl"])

reflections = flatten_reflections(params.input.reflections)
experiments = flatten_experiments(params.input.experiments)

# poly_refined.refl contains strong spots only
# predicted.refl contains predicted spots, but no bounding boxes
# we need to add our own bbox column to predicted.refl

# Code to generate custom sized shoeboxes

reflections = reflections[0]

# Get detector from first experiment (assumes identical detector across experiment)
imageset = experiments[0].imageset
detector = imageset.get_detector()

# image size
dx_det, dy_det = detector[0].get_image_size()

# get reflections x,y,z centroids
x, y, z = reflections["xyzcal.px"].parts()


# bbox params
nx = 10
ny = 10
nz = 1

# Round centroid coordinates to integers
x = flex.floor(x).iround()
y = flex.floor(y).iround()
z = flex.floor(z).iround()

# frames
frame0, frame1 = 0, 1

# initiate empty bounding boxes
bbox = flex.int6(len(reflections))

# construct bounding boxes around centroids
for j, (_x, _y, _z) in enumerate(zip(x, y, z, strict=False)):
    x0_full = _x - nx
    x1_full = _x + nx + 1
    y0_full = _y - ny
    y1_full = _y + ny + 1
    z0_full = _z - nz
    z1_full = _z + nz + 1

    fw_x = x1_full - x0_full
    fw_y = y1_full - y0_full
    fw_z = z1_full - z0_full

    if x0_full < 0:
        # left edge is outside detector: shift right
        x_shift = -x0_full
        x0 = 0
        x1 = min(dx_det, x0 + fw_x)
        if x1 - x0 < fw_x:
            x1 = min(dx_det, x0_full + fw_x)
    elif x1_full >= dx_det:
        x_shift = dx_det - x1_full
        x1 = dx_det
        x0 = max(0, x1 - fw_x)
        if x1 - x0 < fw_x:
            x0 = max(0, x1_full - fw_x)
    else:
        x0 = x0_full
        x1 = x1_full

    if y0_full < 0:
        # left edge is outside detector: shift right
        y_shift = -y0_full
        y0 = 0
        y1 = min(dy_det, y0 + fw_y)
        if y1 - y0 < fw_y:
            y1 = min(dy_det, y0_full + fw_y)
    elif y1_full >= dy_det:
        y_shift = dy_det - y1_full
        y1 = dy_det
        y0 = max(0, y1 - fw_y)
        if y1 - y0 < fw_y:
            y0 = max(0, y1_full - fw_y)
    else:
        y0 = y0_full
        y1 = y1_full

    if z0_full < frame0:
        z_shift = frame0 - z0_full
        z0 = frame0
        z1 = min(frame1, z0 + fw_z)
        if z1 - z0 < fw_z:
            z1 = min(frame1, z0_full + fw_z)
    elif z1_full >= frame1:
        z_shift = frame1 - z1_full
        z1 = frame1
        z0 = max(frame0, z1 - fw_z)
        if z1 - z0 < fw_z:
            z0 = max(frame0, z1_full - fw_z)
    else:
        z0 = z0_full
        z1 = z1_full

    bbox[j] = (x0, x1, y0, y1, z0, z1)

# insert new bboxes
reflections["bbox"] = bbox

final_refls = flex.reflection_table()

modified_counts = 0
idx = 0
all_masks = []
all_counts = []
all_reference = []
for experiment, indices in reflections.iterate_experiments_and_indices(experiments):
    print(f'processing tbl: {idx}')
    imageset = experiment.imageset
    detector = imageset.get_detector()
    subset = reflections.select(indices)
    subset["shoebox"] = flex.shoebox(subset["panel"], subset["bbox"], allocate=True)
    subset.extract_shoeboxes(imageset)
    #final_refls.extend(subset)
#
#    modified = subset["shoebox"].mask_neighbouring(
#        subset["miller_index"],
#        experiment.beam,
#        experiment.detector,
#        Goniometer(),
#        Scan(),
#        experiment.crystal,
#    )
#   modified_counts += modified.count(True)
    #subset.as_file(f'./shoeboxes/tbl_{idx}.refl')
    idx+=1

    x, y, z = subset["xyzcal.px"].parts()
    centroids = np.vstack([x.as_numpy_array(), y.as_numpy_array()]).T

    shoebox_radius = 10
    kdtree = KDTree(centroids)


    masks = []
    counts = []
    for i, refl in enumerate(subset["shoebox"]):
        cx, cy = centroids[i]
        x_, y_, z_ = refl.coords().parts()
        coords = np.vstack([x_.as_numpy_array(), y_.as_numpy_array()]).T
        nearest_indices = kdtree.query(coords)[1]  # indexes into neighbor indices
        nearest_indices = nearest_indices.reshape(
            (2 * shoebox_radius + 1, 2 * shoebox_radius + 1)
        )
        masks.append(
            ((nearest_indices == i) * 1).ravel()
        )  # 0 where neighbor, 1 where belongs to current refl
        counts.append(refl.data.as_numpy_array().ravel())


    r = torch.tensor(np.concatenate(
        [
            subset["background.sum.value"].as_numpy_array().reshape(-1,1), #0
            subset["background.sum.variance"].as_numpy_array().reshape(-1,1), #1
            subset["imageset_id"].as_numpy_array().reshape(-1,1), #2
            subset["intensity.sum.value"].as_numpy_array().reshape(-1,1), #3
            subset["intensity.sum.variance"].as_numpy_array().reshape(-1,1),#4
            subset["miller_index"].as_vec3_double().as_numpy_array(), #5,6,7
            subset["wavelength"].as_numpy_array().reshape(-1,1),#8
            subset["xyzcal.px"].as_numpy_array(), #9,10,11
        ],
        axis=-1,

    ))

    m = torch.tensor(masks)
    c = torch.tensor(counts)
    torch.save(m,f'./shoeboxes/masks_{idx}.pt')
    torch.save(c,f'./shoeboxes/counts_{idx}.pt')
    torch.save(r,f'./shoeboxes/reference_{idx}.pt')
    all_masks.append(m)
    all_counts.append(c)
    all_reference.append(r)

    del subset['shoebox']
    del subset
    gc.collect()

combined_masks = torch.cat(all_masks,dim=0)
combined_counts = torch.cat(all_counts,dim=0)
combined_reference = torch.cat(all_reference,dim=0)

torch.save(combined_masks,'./shoeboxes/combined_masks.pt')
torch.save(combined_counts,'./shoeboxes/combined_counts.pt')
torch.save(combined_reference,'./shoeboxes/combined_reference.pt')
