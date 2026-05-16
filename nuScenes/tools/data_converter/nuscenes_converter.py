# Sourced from: https://github.com/xiaomi-research/unidrivevla/blob/main/nuScenes/tools/data_converter/nuscenes_converter.py
# Handles both v1.0-trainval and v1.0-mini (nuScenes 10-scene subset)
import torch
import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

import mmcv

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global

from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw   = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def geom2anno(map_geoms):
    MAP_CLASSES = ('ped_crossing', 'divider', 'boundary')
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                line = np.array(geom.coords)
                vectors[label].append(line)
    return vectors


def create_nuscenes_infos(root_path, out_path, can_bus_root_path, info_prefix,
                          version='v1.0-trainval', max_sweeps=10, roi_size=(30, 60)):
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes   = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes   = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train   # 7 scenes
        val_scenes   = splits.mini_val     # 3 scenes
        out_path     = osp.join(out_path, 'mini')
    else:
        raise ValueError('unknown version')
    os.makedirs(out_path, exist_ok=True)

    available_scenes      = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes   = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token']
                        for s in train_scenes])
    val_scenes   = set([available_scenes[available_scene_names.index(s)]['token']
                        for s in val_scenes])

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, val scene: {len(val_scenes)}')

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_map_extractor, nusc_can_bus, train_scenes, val_scenes,
        test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print(f'test sample: {len(train_nusc_infos)}')
        data = dict(infos=train_nusc_infos, metadata=metadata)
        mmcv.dump(data, osp.join(out_path, f'{info_prefix}_infos_test.pkl'))
    else:
        print(f'train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}')
        data = dict(infos=train_nusc_infos, metadata=metadata)
        mmcv.dump(data, osp.join(out_path, f'{info_prefix}_infos_train.pkl'))
        data['infos'] = val_nusc_infos
        mmcv.dump(data, osp.join(out_path, f'{info_prefix}_infos_val.pkl'))


def get_available_scenes(nusc):
    available_scenes = []
    print(f'total scene num: {len(nusc.scene)}')
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec   = nusc.get('scene', scene_token)
        sample_rec  = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec      = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print(f'exist scene num: {len(available_scenes)}')
    return available_scenes


def _fill_trainval_infos(nusc, nusc_map_extractor, nusc_can_bus,
                         train_scenes, val_scenes, test=False,
                         max_sweeps=10, fut_ts=12, ego_fut_ts=6):
    train_nusc_infos = []
    val_nusc_infos   = []
    cat2idx = {dic['name']: idx for idx, dic in enumerate(nusc.category)}
    predict_helper = PredictHelper(nusc)

    for sample in mmcv.track_iter_progress(nusc.sample):
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token  = sample['data']['LIDAR_TOP']
        sd_rec       = nusc.get('sample_data', lidar_token)
        cs_record    = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record  = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'map_location': map_location,
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # map annotations
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3]  = np.array(info['lidar2ego_translation'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3]  = np.array(info['ego2global_translation'])
        lidar2global = ego2global @ lidar2ego
        translation = list(lidar2global[:3, 3])
        rotation    = list(Quaternion(matrix=lidar2global).q)
        map_geoms   = nusc_map_extractor.get_map_geom(map_location, translation, rotation)
        info['map_annos'] = geom2anno(map_geoms)

        # 6-camera info
        camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam in camera_types:
            cam_token = sample['data'][cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # sweeps
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
            locs     = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims     = np.array([b.wlh    for b in boxes]).reshape(-1, 3)
            rots     = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations],
                dtype=bool).reshape(-1)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            instance_inds = [nusc.getind('instance', anno['instance_token']) for anno in annotations]

            # agent future trajectories
            num_box = len(boxes)
            gt_fut_trajs  = np.zeros((num_box, fut_ts, 2))
            gt_fut_masks  = np.zeros((num_box, fut_ts))
            for i, anno in enumerate(annotations):
                instance_token = anno['instance_token']
                fut_traj_local = predict_helper.get_future_for_agent(
                    instance_token, sample['token'], seconds=fut_ts / 2, in_agent_frame=True)
                if fut_traj_local.shape[0] > 0:
                    b = boxes[i]
                    fut_traj_scene = convert_local_coords_to_global(
                        fut_traj_local, b.center, Quaternion(matrix=b.rotation_matrix))
                    valid_step = fut_traj_scene.shape[0]
                    gt_fut_trajs[i, 0]          = fut_traj_scene[0] - b.center[:2]
                    gt_fut_trajs[i, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1]
                    gt_fut_masks[i, :valid_step]  = 1

            # ego future trajectory
            ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3))
            ego_fut_masks = np.zeros((ego_fut_ts + 1))
            sample_cur    = sample
            ego_status    = get_ego_status(nusc, nusc_can_bus, sample_cur)
            for i in range(ego_fut_ts + 1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc)
                ego_fut_trajs[i] = pose_mat[:3, 3]
                ego_fut_masks[i] = 1
                if sample_cur['next'] == '':
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            ego_fut_trajs -= np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            ego_fut_trajs -= np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            if ego_fut_trajs[-1][0] >= 2:
                command = np.array([1, 0, 0])   # Turn Right
            elif ego_fut_trajs[-1][0] <= -2:
                command = np.array([0, 1, 0])   # Turn Left
            else:
                command = np.array([0, 0, 1])   # Go Straight
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

            info['gt_boxes']             = gt_boxes
            info['gt_names']             = names
            info['gt_velocity']          = velocity.reshape(-1, 2)
            info['num_lidar_pts']        = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts']        = np.array([a['num_radar_pts'] for a in annotations])
            info['valid_flag']           = valid_flag
            info['instance_inds']        = instance_inds
            info['gt_agent_fut_trajs']   = gt_fut_trajs.astype(np.float32)
            info['gt_agent_fut_masks']   = gt_fut_masks.astype(np.float32)
            info['gt_ego_fut_trajs']     = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks']     = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd']       = command.astype(np.float32)
            info['ego_status']           = ego_status

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def get_ego_status(nusc, nusc_can_bus, sample):
    ego_status = []
    ref_scene = nusc.get('scene', sample['scene_token'])
    try:
        pose_msgs  = nusc_can_bus.get_messages(ref_scene['name'], 'pose')
        steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
        pose_uts   = [msg['utime'] for msg in pose_msgs]
        steer_uts  = [msg['utime'] for msg in steer_msgs]
        ref_utime  = sample['timestamp']
        pose_data  = pose_msgs[locate_message(pose_uts, ref_utime)]
        steer_data = steer_msgs[locate_message(steer_uts, ref_utime)]
        ego_status.extend(pose_data['accel'])
        ego_status.extend(pose_data['rotation_rate'])
        ego_status.extend(pose_data['vel'])
        ego_status.append(steer_data['value'])
    except Exception:
        ego_status = [0] * 10
    return np.array(ego_status).astype(np.float32)


def get_global_sensor_pose(rec, nusc):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
    cs_record   = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
    ego2global  = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=False)
    sensor2ego  = transform_matrix(cs_record['translation'],   Quaternion(cs_record['rotation']),   inverse=False)
    return ego2global.dot(sensor2ego)


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
    sd_rec    = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path   = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
    }
    l2e_r_s     = sweep['sensor2ego_rotation']
    l2e_t_s     = sweep['sensor2ego_translation']
    e2g_r_s     = sweep['ego2global_rotation']
    e2g_t_s     = sweep['ego2global_translation']
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + \
         l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation']    = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def nuscenes_data_prep(root_path, can_bus_root_path, info_prefix, version,
                       dataset_name, out_dir, max_sweeps=10):
    create_nuscenes_infos(root_path, out_dir, can_bus_root_path, info_prefix,
                          version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument('--root-path', type=str, default='./data/nuscenes')
parser.add_argument('--canbus',    type=str, default='./data')
parser.add_argument('--version',   type=str, default='v1.0',
                    help='v1.0 (trainval) or v1.0-mini')
parser.add_argument('--max-sweeps', type=int, default=10)
parser.add_argument('--out-dir',   type=str, default='./data/infos')
parser.add_argument('--extra-tag', type=str, default='nuscenes')
parser.add_argument('--workers',   type=int, default=4)
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=f'{args.version}-trainval',
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        # nuScenes mini: 10 scenes (7 train + 3 val)
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version='v1.0-mini',
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
