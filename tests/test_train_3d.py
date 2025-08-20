from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CONFIGS_DIR = HERE / "configs"


def test_fixture_exists():
    assert (DATA_DIR / "2d/concentration_2d.pt").exists()
    print(DATA_DIR / "2d/concentration_2d.pt")

    assert (CONFIGS_DIR / "config3d.yaml").exists()
    print(CONFIGS_DIR / "config3d.yaml")
