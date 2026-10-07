from pathlib import Path
from dxtbx.model.experiment_list import ExperimentListFactory

EXPT = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/outputs/r0269/018_rg070/out/idx-data_01418_integrated.expt")

def show_obj(name, obj):
    print(f"\n{name}")
    print("  type:", type(obj))
    print("  repr:", repr(obj))

    for attr in [
        "_reader",
        "_format_class",
        "format_class",
        "_format_kwargs",
        "kwargs",
        "_filename",
        "_paths",
        "_indices",
        "_image_range",
    ]:
        if hasattr(obj, attr):
            try:
                print(f"  {attr}:", getattr(obj, attr))
            except Exception as e:
                print(f"  {attr}: <error {type(e).__name__}: {e}>")

def inspect(check_format):
    print("\n" + "=" * 80)
    print("check_format =", check_format)

    experiments = ExperimentListFactory.from_json_file(
        str(EXPT),
        check_format=check_format,
    )

    print("num experiments:", len(experiments))
    print("num imagesets:", len(experiments.imagesets()))

    exp = experiments[0]
    imageset = experiments.imagesets()[0]

    show_obj("experiment", exp)
    show_obj("imageset", imageset)

    try:
        reader = imageset.reader()
        show_obj("imageset.reader()", reader)
    except Exception as e:
        print("imageset.reader() failed:", type(e).__name__, e)

    try:
        raw = imageset.get_raw_data(0)
        print("\nget_raw_data(0): OK")
        print("  raw type:", type(raw))
        print("  num panels:", len(raw))
        for i in range(min(3, len(raw))):
            panel = raw[i]
            arr = panel.as_numpy_array()
            print(f"  panel {i}: flex shape={panel.all()}, numpy shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print("\nget_raw_data(0): FAILED")
        print("  error:", type(e).__name__, e)

for check_format in [True, False]:
    inspect(check_format)
