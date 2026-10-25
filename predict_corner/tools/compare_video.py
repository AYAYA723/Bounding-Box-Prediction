import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-label", default="Before: original weights / same corrected ROI")
    parser.add_argument("--after-label", default="After: 1000 synthetic training images")
    args = parser.parse_args()
    before = cv2.VideoCapture(str(args.before))
    after = cv2.VideoCapture(str(args.after))
    count = int(before.get(cv2.CAP_PROP_FRAME_COUNT))
    assert count == int(after.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = before.get(cv2.CAP_PROP_FPS)
    assert abs(fps - after.get(cv2.CAP_PROP_FPS)) < .001
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 526))
    assert writer.isOpened()
    for index in range(count):
        success_a, a = before.read()
        success_b, b = after.read()
        assert success_a and success_b
        display = np.zeros((526, 1280, 3), np.uint8)
        display[40:, :640] = cv2.resize(a, (640, 486))
        display[40:, 640:] = cv2.resize(b, (640, 486))
        cv2.putText(display, args.before_label, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.putText(display, args.after_label, (650, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        writer.write(display)
    before.release()
    after.release()
    writer.release()
    check = cv2.VideoCapture(str(args.output))
    assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    check.release()
    print(f"Saved {count} synchronized frames: {args.output}")


if __name__ == "__main__":
    main()
