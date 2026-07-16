import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_submit_module_cli_creates_valid_atomic_zip(tmp_path: Path):
    data_root = tmp_path / "data"
    scene = data_root / "scene_a"
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "train" / "sparse" / "0").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(scene / "train" / "images" / "train.JPG", format="JPEG")
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,0,0,0,10,10,4,3,8,6\n",
        encoding="utf-8",
    )
    submission = tmp_path / "outputs" / "candidate" / "rendered" / "scene_a"
    submission.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(submission / "target.JPG", format="JPEG")
    output = tmp_path / "submission.zip"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bts_nvs.submit",
            "--data-root",
            str(data_root),
            "--submission",
            str(submission.parent),
            "--out",
            str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "sha256=" in completed.stdout
