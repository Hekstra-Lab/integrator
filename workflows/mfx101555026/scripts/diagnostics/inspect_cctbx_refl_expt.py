from pathlib import Path
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory

folder = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx/scale_r0269_018_p212121"
)

refl_path = folder / "mfx_r0269_018_scale.refl"
expt_path = folder / "mfx_r0269_018_scale.expt"

refl = flex.reflection_table.from_file(str(refl_path))
expts = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)

print("reflection rows:", len(refl))
print("experiment count:", len(expts))

print("\nColumns:")
for key in refl.keys():
    print(" ", key)

print("\nFirst 10 scaled reflection rows:")
for i in range(10):
    print("row", i)
    print("  id:", refl["id"][i])
    print("  hkl_asym:", refl["miller_index_asymmetric"][i])
    print("  I:", refl["intensity.sum.value"][i])
    print("  var:", refl["intensity.sum.variance"][i])

print("\nFirst 10 experiments:")
for i in range(min(10, len(expts))):
    exp = expts[i]
    print("\nexperiment", i)
    print("  identifier:", exp.identifier)

    try:
        paths = exp.imageset.paths()
        print("  paths:", paths[:3])
    except Exception as e:
        print("  paths: ERROR", e)

    try:
        indices = exp.imageset.indices()
        print("  indices:", indices[:10])
    except Exception as e:
        print("  indices: ERROR", e)