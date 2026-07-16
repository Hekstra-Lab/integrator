from pathlib import Path
import time
import numpy as np

from dxtbx.model.experiment_list import ExperimentListFactory


BASE = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")
IN_DIR = BASE / "outputs/r0269/018_rg070/out"
OUT_DIR = BASE / "raw_panel_cache_r0269_018_rg070"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_NUM = 15833

expt_path = IN_DIR / f"idx-data_{IMAGE_NUM:05d}_integrated.expt"
out_path = OUT_DIR / f"raw_panels_image_{IMAGE_NUM:05d}.npz"

print("expt:", expt_path)
print("out: ", out_path)

if not expt_path.exists():
    raise FileNotFoundError(expt_path)

t0 = time.perf_counter()
experiments = ExperimentListFactory.from_json_file(
    str(expt_path),
    check_format=True,
)
t_load = time.perf_counter() - t0

t0 = time.perf_counter()
raw = experiments[0].imageset.get_raw_data(0)
t_raw = time.perf_counter() - t0

panels = {
    f"panel_{i:02d}": raw[i].as_numpy_array().astype(np.float32, copy=False)
    for i in range(len(raw))
}

t0 = time.perf_counter()
np.savez_compressed(out_path, **panels)
t_save = time.perf_counter() - t0

try:
    experiments[0].imageset.clear_cache()
except Exception:
    pass

size_mb = out_path.stat().st_size / 1024 / 1024

print(f"saved: {out_path}")
print(f"size: {size_mb:.1f} MB")
print(f"load .expt:   {t_load:.2f} sec")
print(f"get_raw_data: {t_raw:.2f} sec")
print(f"save .npz:    {t_save:.2f} sec")
print(f"total:        {t_load + t_raw + t_save:.2f} sec")
