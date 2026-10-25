import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--columns', type=int, required=True, help='Number of inner corners across the chessboard')
    parser.add_argument('--rows', type=int, required=True, help='Number of inner corners down the chessboard')
    parser.add_argument('--square-mm', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory to preserve the previous calibration.')
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    pattern = (args.columns, args.rows)
    points = np.zeros((args.rows * args.columns, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[:args.columns, :args.rows].T.reshape(-1, 2) * args.square_mm
    objects, image_points, accepted, rejected = [], [], [], []
    image_size = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = (gray.shape[1], gray.shape[0])
        if image_size is not None and size != image_size:
            parser.error('All calibration images must use the same original resolution.')
        image_size = size
        success, corners = cv2.findChessboardCornersSB(gray, pattern)
        if success:
            objects.append(points.copy())
            image_points.append(corners)
            accepted.append(path)
        else:
            rejected.append(str(path))
    if len(accepted) < 10:
        parser.error(f'Only {len(accepted)} usable chessboard views; provide at least 10 varied views.')
    rms, K, distortion, rvecs, tvecs = cv2.calibrateCamera(objects, image_points, image_size, None, None)
    per_view = []
    args.output.mkdir(parents=True)
    for path, corners, rvec, tvec in zip(accepted, image_points, rvecs, tvecs):
        projected = cv2.projectPoints(points, rvec, tvec, K, distortion)[0]
        error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=2))))
        per_view.append({'image': str(path), 'rms_px': error,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        preview = cv2.imread(str(path))
        cv2.drawChessboardCorners(preview, pattern, corners, True)
        cv2.imwrite(str(args.output / path.name), preview)
    camera = {'model': 'pinhole', 'calibration_kind': 'measured',
              'width': image_size[0], 'height': image_size[1], 'cam_K': K.tolist(),
              'dist_coeffs': distortion.reshape(-1).tolist(), 'rms_reprojection_px': float(rms),
              'inner_corners': list(pattern), 'square_mm': args.square_mm,
              'used_images': len(accepted), 'per_view': per_view, 'rejected_images': rejected}
    (args.output / 'camera.json').write_text(json.dumps(camera, indent=2) + '\n')
    print(json.dumps({key: value for key, value in camera.items() if key not in {'per_view', 'rejected_images'}}, indent=2))


if __name__ == '__main__':
    main()
