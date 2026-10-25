import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from predict_corner.data_prep.prepare_video_crops import camera_body_boxes, camera_rois, load_annotations, overlap_fraction


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = load_annotations()
    configurations = {'original': base, 'adaptive': dict(base, roi_mode='adaptive')}
    summaries = {}
    for name, annotations in configurations.items():
        widths, other, own = [], [], []
        for frame_id in range(annotations['frame_count']):
            bodies = camera_body_boxes(annotations, frame_id)
            crops = camera_rois(annotations, frame_id)
            for side, crop in enumerate(crops):
                widths.append(int(crop[2] - crop[0]))
                other.append(overlap_fraction(crop, bodies[1 - side]))
                own.append(overlap_fraction(crop, bodies[side]))
        summaries[name] = {'target_crops': len(widths), 'crop_width_range_original_px': [min(widths), max(widths)],
                           'median_crop_width_original_px': float(np.median(widths)),
                           'other_camera_at_least_half_inside_percent': float(np.mean(np.array(other) >= .5) * 100),
                           'other_camera_more_than_10pct_inside_percent': float(np.mean(np.array(other) > .1) * 100),
                           'own_annotated_body_fully_inside_percent': float(np.mean(np.array(own) >= .999) * 100)}
    (args.output / 'geometry.json').write_text(json.dumps(summaries, indent=2) + '\n')
    frames_root = root / 'datasets/action4/video_test_24'
    frames = json.loads((frames_root / 'frames.json').read_text())['frames']
    sheet = Image.new('RGB', (2400, 6 * 480), 'white')
    for entry in frames:
        with Image.open(frames_root / entry['file']) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for color, name in [('#ffb030', 'original'), ('#30eaff', 'adaptive')]:
            for roi in camera_rois(configurations[name], entry['frame_id']):
                draw.rectangle(roi.tolist(), outline=color, width=6)
        image.thumbnail((600, 455))
        index = entry['index']
        left, top = index % 4 * 600, index // 4 * 480
        sheet.paste(image, (left, top))
        ImageDraw.Draw(sheet).text((left + 4, top + 457), f'{entry["time_seconds"]:.2f}s / orange: original / cyan: adaptive', fill='black')
    sheet.save(args.output / 'roi_comparison.jpg', quality=94)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
