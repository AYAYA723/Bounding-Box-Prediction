import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def detect_tags(video, step, minimum_side):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), parameters)
    capture = cv2.VideoCapture(str(video))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    observations = []
    for frame_id in range(0, count, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = capture.read()
        assert success, frame_id
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        for marker, identity in zip(corners, ids.flatten()):
            points = marker.reshape(4, 2)
            sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
            if int(identity) in [18, 19] and sides.min() >= minimum_side:
                observations.append({'frame_id': frame_id, 'time_seconds': frame_id / fps,
                                     'tag_id': int(identity), 'corners_xy': points.tolist(),
                                     'minimum_side_px': float(sides.min())})
        if frame_id % (step * 20) == 0:
            print(f'Detection frame={frame_id}/{count} accepted={len(observations)}', flush=True)
    capture.release()
    return observations, size, fps, count


def calibrate(observations, size, model, centered=False):
    square = np.float32([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    objects = [square.copy() for _ in observations]
    images = [np.asarray(row['corners_xy'], dtype=np.float32) for row in observations]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if model == 'pinhole':
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    elif model == 'radial1':
        flags |= cv2.CALIB_FIX_K2
    if centered:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    result = cv2.calibrateCameraExtended(objects, images, size, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9))
    rms, K, distortion, rvecs, tvecs, intrinsics_std, _, per_view = result
    return {'model_name': model, 'principal_point_fixed': centered, 'rms_px': float(rms),
            'cam_K': K.tolist(), 'dist_coeffs': distortion.reshape(-1).tolist(),
            'intrinsics_std': intrinsics_std.reshape(-1).tolist(),
            'per_view_rms_px': per_view.reshape(-1).tolist(), 'views': len(observations)}


def heldout_error(observations, calibration):
    square = np.float64([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    K = np.array(calibration['cam_K'])
    distortion = np.array(calibration['dist_coeffs'])
    errors = []
    for row in observations:
        points = np.array(row['corners_xy'])
        success, rvec, tvec = cv2.solvePnP(square, points, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        assert success
        projected = cv2.projectPoints(square, rvec, tvec, K, distortion)[0].reshape(4, 2)
        errors.append(float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1)))))
    return float(np.median(errors)), float(np.mean(errors))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, default=root / 'head_left_rgb_raw.mp4')
    parser.add_argument('--step', type=int, default=15)
    parser.add_argument('--minimum-side', type=float, default=45)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.observations:
        data = json.loads(args.observations.read_text())
        observations, size = data['observations'], tuple(data['image_size_wh'])
    else:
        observations, size, fps, count = detect_tags(args.video, args.step, args.minimum_side)
        data = {'video': str(args.video), 'video_sha256': hashlib.sha256(args.video.read_bytes()).hexdigest(),
                'image_size_wh': size, 'fps': fps, 'video_frames': count, 'step': args.step,
                'dictionary': 'DICT_APRILTAG_36h11', 'tag_ids': [18, 19], 'observations': observations}
    (args.output / 'observations.json').write_text(json.dumps(data, indent=2) + '\n')
    assert len(observations) >= 20
    models = {}
    for centered in [False, True]:
        for name in ['pinhole', 'radial1', 'radial2']:
            label = name + ('_centered' if centered else '_free_center')
            model = calibrate(observations, size, name, centered)
            train = [row for row in observations if row['frame_id'] // args.step % 4 != 0]
            test = [row for row in observations if row['frame_id'] // args.step % 4 == 0]
            fold = calibrate(train, size, name, centered)
            model['heldout_median_rms_px'], model['heldout_mean_rms_px'] = heldout_error(test, fold)
            model['heldout_fit_K'] = fold['cam_K']
            models[label] = model
            (args.output / 'candidate_models.json').write_text(json.dumps(models, indent=2) + '\n')
            print(label, f"rms={model['rms_px']:.3f}, heldout={model['heldout_mean_rms_px']:.3f}",
                  model['cam_K'], model['dist_coeffs'], flush=True)
    print(f'COMPLETE observations={len(observations)} seconds={time.perf_counter()-started:.1f}', flush=True)


if __name__ == '__main__':
    main()
