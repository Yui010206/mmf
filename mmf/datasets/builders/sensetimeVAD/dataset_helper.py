import os
import numpy as np
import logging
import random
from collections import defaultdict
from PIL import Image
from utils.utils import extend_by_x
import json


HUMAN_POWERED_TRANSPORT = 1507442
PEDESTRIAN = 221488
THREE_WHEELER = 739288
MOTORCYCLE = 34493
BICYCLE = 11442
TRICYCLE = 536135
CYCLIST = 2125610
CYCLING = 53121
MANNEQUIN = 855680

ACT_LABEL_MAPPING = {
    "tumble": 1,
    "tumble_start": 1,
    "tumble_end": 1,
    "climb": 1,
    "climbing": 1,
    "leaflet": 1,
    "wallpollution": 1,
    "geton": 1,
    "getoff": 1,
    "throwing": 1,
    "kicking": 1,
    "squat": 0,
    "squat_start": 0,
    "squat_end": 0,
    "normal": 0
}

def reading_with_exception_handling(record, image_tmpl, id_):
    "reading data with exception handling"
    img_filename = os.path.join(
        record.path, image_tmpl.format(id_)
    )
    indice = id_
    while not (os.path.exists(img_filename) and os.path.getsize(img_filename)):
        id_ += 1
        img_filename = os.path.join(
            record.path, image_tmpl.format(id_)
        )
        if id_ > record.end:
            id_ = 0
        if id_ == indice - 1:
            assert os.path.exists(img_filename), \
                'out of index in {}'.format(img_filename)
    return img_filename
    

def load_and_filter_rois(tracklet_info, frame_idx, scan_interval, bbox_expand_ratio,
                         filter_motor, use_mannequin, ignore_squat, size):
    """
    Load rois of certain frame index information and filter unused rois.
    Params:
        tracklet_info: dict([list]):
            key: frame_index;
            value: tracklets of the correspoinding frame.
        frame_idx: int, index of the specific frame
        size： tuple, size of the frame, (width, height)
    Returns:
        bbox_data_final: list. Sorted bbox information for the dataset.
        [track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            type_label, action_flag, action_label]
    """
    areas = []
    bbox_data = []
    motor_data = []
    motor_ped_data = []
    bbox_data_final = []

    if len(tracklet_info) == 0:
        return []
    # get the matching tracklet index.
    shift = 0
    rst = None
    while shift < scan_interval:
        right_idx = frame_idx + shift
        rst = tracklet_info.get(right_idx, None)
        if rst is not None:
            break
        left_idx = frame_idx - shift
        rst = tracklet_info.get(left_idx, None)
        if rst is not None:
            break
        shift += 1
    if rst is None:
        return []

    for bbox in rst:
        track_id = int(bbox[0])
        bbox_x = float(bbox[1])
        bbox_y = float(bbox[2])
        bbox_w = float(bbox[3])
        bbox_h = float(bbox[4])
        type_label = int(bbox[5])
        action_flag = int(bbox[7])
        if action_flag > 0:
            action_label = bbox[8]
        else:
            action_label = -1
        "skip illegal rois"
        if bbox_w <= 0 or bbox_h <= 0 or bbox_x < 0 or bbox_y < 0:
            continue
        "whether to expand bbox"
        if bbox_expand_ratio != 1.0:
            half_ratio = (bbox_expand_ratio - 1) / 2.0
            bbox_x1 = np.maximum(
                0, bbox_x - half_ratio * bbox_w + 1)
            bbox_x2 = np.minimum(
                size[0], bbox_x1 + bbox_expand_ratio * bbox_w - 1)
            bbox_y1 = np.maximum(
                0, bbox_y - half_ratio * bbox_h + 1)
            bbox_y2 = np.minimum(
                size[1], bbox_y1 + bbox_expand_ratio * bbox_h - 1)
        elif bbox_expand_ratio == 1.0:
            bbox_x1 = bbox_x
            bbox_x2 = np.minimum(size[0], bbox_x + bbox_w)
            bbox_y1 = bbox_y
            bbox_y2 = np.minimum(size[1], bbox_y + bbox_h)
        "whether to filter the motor"
        if type_label == PEDESTRIAN or type_label == MANNEQUIN:
            bbox_data.append(
                [track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, type_label,
                    action_flag, action_label]
            )
        if filter_motor:
            if type_label == HUMAN_POWERED_TRANSPORT or \
                    type_label == THREE_WHEELER or \
                    type_label == MOTORCYCLE or \
                    type_label == BICYCLE or \
                    type_label == TRICYCLE:
                motor_data.append(
                    [track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                )
            if type_label == CYCLIST or type_label == CYCLING:
                motor_ped_data.append(
                    [track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                )

    "filter out pedestrian bbox which contain the motor."
    if filter_motor:
        for bbox_ in bbox_data:
            ignore_flag = False
            for motor_bbox in motor_data:
                if abs((bbox_[1] + bbox_[3]) / 2 - (motor_bbox[1] + motor_bbox[3]) / 2) < size[0] * 0.01:
                    ignore_flag = True
                    break
            for motor_ped_bbox in motor_data:
                if abs((bbox_[1] + bbox_[3]) / 2 - (motor_ped_bbox[1] + motor_ped_bbox[3]) / 2) < size[0] * 0.01:
                    ignore_flag = True
                    break
            if ignore_flag:
                continue
            else:
                bbox_data_final.append(bbox_)
    else:
        bbox_data_final = bbox_data

    "mannequin and action label"
    for bbox_ in bbox_data_final:
        if use_mannequin:
            bbox_[-3] = 1
        else:
            if bbox_[-3] == MANNEQUIN:
                bbox_[-3] = -1
            else:
                bbox_[-3] = 1
        if ignore_squat:
            ACT_LABEL_MAPPING['squat'] = -1
            ACT_LABEL_MAPPING['squat_start'] = -1
            ACT_LABEL_MAPPING['squat_end'] = -1
        if bbox_[-2] > 0:
            bbox_[-1] = ACT_LABEL_MAPPING[bbox_[-1]]

    "sort with area"
    if len(bbox_data_final) > 1:
        for bbox_ in bbox_data_final:
            areas.append((bbox_[3] - bbox_[1]) * (bbox_[4] - bbox_[2]))
        areas = np.array(areas)
        sorted_idx = np.argsort(areas)[::-1]
        bbox_data_final = [bbox_data_final[i] for i in sorted_idx]

    return bbox_data_final

def cal_iou(roi_A, roi_B):
    s_A = (roi_A[2] - roi_A[0]) * (roi_A[3] - roi_A[1])
    s_B = (roi_B[2] - roi_B[0]) * (roi_B[3] - roi_B[1])

    if s_A <= 0 or s_B <= 0:
        return 0
    else:
        inter_tx = max(roi_A[0], roi_B[0])
        inter_bx = min(roi_A[2], roi_B[2])
        inter_ty = max(roi_A[1], roi_B[1])
        inter_by = min(roi_A[3], roi_B[3])

        intersect = max(
            0., (inter_bx - inter_tx)) * max(0., (inter_by - inter_ty))
        return float(intersect) / (s_A + s_B - intersect)
