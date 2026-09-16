from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_requirements_install_local_televuer_with_haptic_dependencies():
    requirements = (ROOT / "requirements.txt").read_text()
    televuer_project = (ROOT / "teleop" / "televuer" / "pyproject.toml").read_text()

    assert "-e ./teleop/televuer" in requirements
    assert '"vuer[all]==0.0.60"' in televuer_project
    assert '"params-proto<3"' in televuer_project


def test_installed_televuer_imports_against_vuer_060():
    assert metadata.version("vuer") == "0.0.60"

    from televuer import TeleVuerWrapper

    assert TeleVuerWrapper.__name__ == "TeleVuerWrapper"
