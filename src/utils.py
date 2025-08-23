from pathlib import Path

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(__file__).absolute().parent

CONFIGS = {}
for v in list(Path(ROOT_DIR).parent.glob("tests/configs/*.yaml")):
    key = v.name.removesuffix(".yaml")
    CONFIGS[key] = v

DATA = {}
for v in list(Path(ROOT_DIR).parent.glob("tests/data/*/")):
    key = v.name
    paths = [p for p in list(v.glob("**/*.pt"))]
    paths.sort()
    keys = [p.name.split("_")[0] for p in paths]
    vals = {}
    for k, p in zip(keys, paths, strict=False):
        vals[k] = p

    DATA[key] = vals
