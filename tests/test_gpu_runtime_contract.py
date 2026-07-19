from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_compose_runs_as_the_host_user() -> None:
    compose = (ROOT / "infra" / "gpu" / "docker-compose.yml").read_text()

    assert 'user: "${HOST_UID:-1000}:${HOST_GID:-1000}"' in compose


def test_gpu_scripts_export_the_host_identity() -> None:
    for script_name in ("smoke_chair.sh", "verify_gpu.sh"):
        script = (ROOT / "infra" / "gpu" / script_name).read_text()
        assert 'export HOST_UID="$(id -u)"' in script
        assert 'export HOST_GID="$(id -g)"' in script
