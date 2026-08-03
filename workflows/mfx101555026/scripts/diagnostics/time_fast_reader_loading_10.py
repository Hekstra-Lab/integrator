from pathlib import Path
import time

from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.format.FormatXTCJungfrau import FormatXTCJungfrau

BASE = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")
IN_DIR = BASE / "outputs/r0269/018_rg070/out"

expt_paths = sorted(IN_DIR.glob("idx-data_*_integrated.expt"))[:10]

print("num files:", len(expt_paths))

total_load = 0.0
total_override = 0.0
total_raw = 0.0

for i, expt_path in enumerate(expt_paths, start=1):
    print()
    print(f"[{i}/{len(expt_paths)}] {expt_path.name}")

    t0 = time.perf_counter()
    experiments = ExperimentListFactory.from_json_file(
        str(expt_path),
        check_format=False,
    )
    total_load += time.perf_counter() - t0

    t0 = time.perf_counter()
    imageset = experiments.imagesets()[0]
    reader = imageset.reader()
    reader.format_class = FormatXTCJungfrau
    total_override += time.perf_counter() - t0

    print("  reader:", type(reader))
    print("  format_class:", getattr(reader, "format_class", None))

    t0 = time.perf_counter()
    raw = imageset.get_raw_data(0)
    total_raw += time.perf_counter() - t0

    print("  raw panels:", len(raw))
    print("  panel 0 shape:", raw[0].all())

    try:
        imageset.clear_cache()
    except Exception:
        pass

print()
print("Timing summary")
print(f"load expt check_format=False: {total_load:.2f} sec")
print(f"override reader format_class: {total_override:.2f} sec")
print(f"get_raw_data total:          {total_raw:.2f} sec")
print(f"accounted total:             {total_load + total_override + total_raw:.2f} sec")
