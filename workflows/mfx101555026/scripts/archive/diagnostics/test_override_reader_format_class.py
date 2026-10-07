from pathlib import Path

from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.format.FormatXTCJungfrau import FormatXTCJungfrau

EXPT = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/outputs/r0269/018_rg070/out/idx-data_01418_integrated.expt")

experiments = ExperimentListFactory.from_json_file(
    str(EXPT),
    check_format=False,
)

imageset = experiments.imagesets()[0]
reader = imageset.reader()

print("before reader:", type(reader))
print("before format_class:", getattr(reader, "format_class", None))

reader.format_class = FormatXTCJungfrau

print("after reader:", type(reader))
print("after format_class:", getattr(reader, "format_class", None))

raw = imageset.get_raw_data(0)

print("raw OK")
print("num panels:", len(raw))
print("panel 0 shape:", raw[0].all())
