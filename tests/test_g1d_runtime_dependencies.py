from pathlib import Path


ROOT = Path(__file__).parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
G1D_DOC = ROOT / "teleop" / "G1D_QUEST_DEX3.md"


def _requirements():
    return {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_g1d_runtime_pins_numpy_1_for_pinocchio_and_a_pyarrow_release_compatible_with_it():
    requirements = _requirements()
    assert "numpy==1.26.4" in requirements
    assert "pyarrow==20.0.0" in requirements


def test_g1d_runtime_documentation_explains_the_numpy_pyarrow_rerun_compatibility_pin():
    doc = G1D_DOC.read_text(encoding="utf-8")
    for text in ("numpy==1.26.4", "pyarrow==20.0.0", "rerun-sdk==0.20.1"):
        assert text in doc
