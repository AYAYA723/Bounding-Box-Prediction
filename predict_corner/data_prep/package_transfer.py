import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / "transfer" / f"Action4_transfer_{stamp}.zip"
    output.parent.mkdir(exist_ok=True)
    tracked = subprocess.check_output(
        ["git", "-c", "core.excludesFile=.git/info/exclude", "ls-files", "-z"], cwd=root,
    ).decode("utf-8").split("\0")
    files = {root / name for name in tracked if name and not name.endswith(".ipynb")}
    files.update((root / "predict_corner").rglob("*.py"))
    files.update((root / "predict_corner").rglob("*.md"))
    files.update((root / "predict_corner/checkpoints").glob("*.pt"))
    files.update((root / "datasets/action4/train_pbr/000000/rgb").glob("*.png"))
    files.update((root / "datasets/action4/video_test_24/frames").glob("*.jpg"))
    files.update((root / "datasets/action4/video_test_24/crops").glob("*.jpg"))
    files.update(path for path in (root / "results").rglob("*")
                 if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".png", ".jpg", ".md", ".log"))
    files.update([
        root / "datasets/action4/video_test_24/frames.json",
        root / "datasets/action4/video_test_24/manual_rois.json",
        root / "datasets/action4/video_test_24/frames_contact_sheet.jpg",
        root / "head_left_rgb_raw.mp4",
        root / "相机资产/0468e0a404ddc50b1663be3cac636dd8.glb",
    ])
    # Include an existing local notebook-free workflow, not Git internals or environments.
    hashes = {}
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for path in sorted(files):
            name = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
            compression = zipfile.ZIP_DEFLATED if path.suffix in (".py", ".md", ".json", ".jsonl", ".csv", ".txt") else zipfile.ZIP_STORED
            archive.write(path, "action4_transfer/" + name, compress_type=compression, compresslevel=1)
        archive.writestr("action4_transfer/README_TRANSFER.md",
                         (root / "predict_corner/data_prep/TRANSFER_README.md").read_bytes())
        archive.writestr("action4_transfer/SHA256SUMS.json",
                         json.dumps(hashes, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        assert bad_member is None, bad_member
        file_count = len(archive.namelist())
    print(json.dumps({"archive": str(output), "file_count": file_count,
                      "size_mib": round(output.stat().st_size / 1024**2, 2),
                      "zip_crc_check": "PASS"}, ensure_ascii=False, indent=2))
    videos_output = output.with_name(f"Action4_full_video_results_{stamp}.zip")
    with zipfile.ZipFile(videos_output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted((root / "results").rglob("*.mp4")):
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(videos_output) as archive:
        assert archive.testzip() is None
    print(json.dumps({"full_video_archive": str(videos_output),
                      "size_mib": round(videos_output.stat().st_size / 1024**2, 2),
                      "zip_crc_check": "PASS"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
