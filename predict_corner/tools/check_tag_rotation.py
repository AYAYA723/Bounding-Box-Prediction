import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def compare_rotation(predictions, observations, camera, end_seconds=3.5):
    records = {row['frame_id']: row for row in predictions}
    K = np.asarray(camera['cam_K'])
    distortion = np.asarray(camera['dist_coeffs'])
    square = np.float64([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    previous, rows = {}, []
    for observation in sorted(observations, key=lambda row: (row['frame_id'], row['tag_id'])):
        if observation['time_seconds'] > end_seconds:
            continue
        side = {18: 'L', 19: 'R'}[observation['tag_id']]
        target = next(row for row in records[observation['frame_id']]['targets'] if row['side'] == side)
        if not target['pose']['success']:
            previous.pop(side, None)
            continue
        success, rvec, _ = cv2.solvePnP(square, np.asarray(observation['corners_xy']), K, distortion,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            continue
        tag_R = cv2.Rodrigues(rvec)[0]
        net_R = np.asarray(target['pose']['R_object_to_camera'])
        if side in previous:
            old_tag, old_net, old_time = previous[side]
            tag_delta = tag_R @ old_tag.T
            net_delta = net_R @ old_net.T
            rows.append({'frame_id': observation['frame_id'], 'side': side,
                         'delta_time_seconds': observation['time_seconds'] - old_time,
                         'tag_rotation_change_deg': rotation_angle(tag_delta),
                         'network_rotation_change_deg': rotation_angle(net_delta),
                         'relative_rotation_disagreement_deg': rotation_angle(net_delta @ tag_delta.T)})
        previous[side] = tag_R, net_R, observation['time_seconds']
    summary = {'interval_seconds': [0, end_seconds], 'rotation_pairs': len(rows),
               'method': 'compare camera-frame relative rotations; fixed marker-to-body rotation cancels',
               'reference_assumptions': 'Tags 18/19 are rigid planar squares on the initial L/R objects; estimated K is used',
               'absolute_object_pose_ground_truth': False}
    for key in ['tag_rotation_change_deg', 'network_rotation_change_deg', 'relative_rotation_disagreement_deg']:
        values = [row[key] for row in rows]
        summary[key] = {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
                        'maximum': float(np.max(values))}
    return {'summary': summary, 'pairs': rows}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--observations', type=Path, default=root / 'results/video_tag_calibration_dense_20261025/observations.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run / 'predictions.jsonl').read_text().splitlines()]
    observations = json.loads(args.observations.read_text())['observations']
    camera = json.loads((args.run / 'run.json').read_text())['camera']
    report = compare_rotation(records, observations, camera)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
