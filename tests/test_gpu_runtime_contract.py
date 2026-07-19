from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_compose_runs_as_the_host_user() -> None:
    compose = (ROOT / "infra" / "gpu" / "docker-compose.yml").read_text()

    assert 'HOST_UID: "${HOST_UID:?HOST_UID must be set}"' in compose
    assert 'HOST_GID: "${HOST_GID:?HOST_GID must be set}"' in compose
    assert (
        'user: "${HOST_UID:?HOST_UID must be set}:${HOST_GID:?HOST_GID must be set}"'
        in compose
    )


def test_gpu_scripts_export_the_host_identity() -> None:
    for script_name in ("smoke_chair.sh", "verify_gpu.sh"):
        script = (ROOT / "infra" / "gpu" / script_name).read_text()
        assert 'export HOST_UID="$(id -u)"' in script
        assert 'export HOST_GID="$(id -g)"' in script


def test_gpu_image_preloads_torch_weights_for_non_root_users() -> None:
    dockerfile = (ROOT / "infra" / "gpu" / "Dockerfile").read_text()

    env_position = dockerfile.index("ENV HOME=/tmp")
    preload_position = dockerfile.index("lpips.LPIPS(net='alex')")

    assert env_position < preload_position
    assert "TORCH_HOME=/opt/bts-nvs/torch-cache" in dockerfile
    assert "chmod -R a+rX /opt/bts-nvs/torch-cache" in dockerfile


def test_gpu_image_registers_the_host_uid_for_torch_inductor() -> None:
    dockerfile = (ROOT / "infra" / "gpu" / "Dockerfile").read_text()
    verifier = (ROOT / "infra" / "gpu" / "verify_gpu.sh").read_text()

    assert "ARG HOST_UID" in dockerfile
    assert "ARG HOST_GID" in dockerfile
    assert 'getent passwd "$HOST_UID"' in dockerfile
    assert "pwd.getpwuid(os.getuid())" in verifier
