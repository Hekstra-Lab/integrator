from pathlib import Path
import time

from dxtbx.model.experiment_list import ExperimentListFactory

EXPT = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/outputs/r0269/018_rg070/out/idx-data_01418_integrated.expt")

def inspect(check_format):
    print()
    print("=" * 80)
    print(f"check_format={check_format}")

    t0 = time.perf_counter()
    experiments = ExperimentListFactory.from_json_file(str(EXPT), check_format=check_format)
    print(f"load expt: {time.perf_counter() - t0:.2f} sec")

    exp = experiments[0]
    imageset = exp.imageset

    print("imageset type:", type(imageset))
    print("detector panels:", len(exp.detector))

    t0 = time.perf_counter()
    raw = imageset.get_raw_data(0)
    print(f"get_raw_data(0): {time.perf_counter() - t0:.2f} sec")

    print("raw type:", type(raw))
    print("number of raw panels:", len(raw))

    for i in range(min(5, len(raw))):
        panel = raw[i]
        arr = panel.as_numpy_array()
        print(f"panel {i}: flex shape={panel.all()}, numpy shape={arr.shape}, dtype={arr.dtype}")

    try:
        imageset.clear_cache()
    except Exception:
        pass

inspect(check_format=True)
inspect(check_format=False)
