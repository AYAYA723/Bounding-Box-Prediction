import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def main():
    project_dir = Path(__file__).resolve().parents[2]
    video_path = project_dir / "head_left_rgb_raw.mp4"
    output_dir = project_dir / "datasets/action4/video_test_24"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 24
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    tile_width = 600
    tile_height = round(height * tile_width / width)
    sheet = Image.new("RGB", (4 * tile_width, 6 * (tile_height + 34)), "#eeeeee")
    frames = []

    for index in range(count):
        frame_id = int((index + 0.5) * frame_count / count)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, bgr = capture.read()
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        filename = f"{index:02d}_frame_{frame_id:06d}.jpg"
        image.save(frames_dir / filename, quality=95)
        left = index % 4 * tile_width
        top = index // 4 * (tile_height + 34)
        sheet.paste(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS), (left, top))
        ImageDraw.Draw(sheet).text(
            (left + 8, top + tile_height + 7),
            f"{index:02d} | frame {frame_id} | {frame_id / fps:.2f} s", fill="black", font=font,
        )
        frames.append({"index": index, "frame_id": frame_id,
                       "time_seconds": frame_id / fps, "file": f"frames/{filename}"})

    capture.release()
    sheet.save(output_dir / "frames_contact_sheet.jpg", quality=94)
    manifest = {"source_video": video_path.name, "fps": fps,
                "frame_count": frame_count, "duration_seconds": frame_count / fps,
                "image_size_wh": [width, height], "sample_count": count,
                "sampling": "center frame of each of 24 equal temporal bins",
                "split": "video test only; not used for training", "frames": frames}
    (output_dir / "frames.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved {count} frames to {output_dir}")
    print(f"Interval: {frame_count / fps / count:.3f} s")


if __name__ == "__main__":
    main()
