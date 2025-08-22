from pathlib import Path

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(__file__).absolute().parent

CONFIGS = {}
for v in list(Path(ROOT_DIR).parent.glob("tests/configs/*.yaml")):
    key = v.name.removesuffix(".yaml")
    CONFIGS[key] = v
