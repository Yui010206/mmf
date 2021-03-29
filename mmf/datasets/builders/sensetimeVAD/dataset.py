import json
import os

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.general import get_mmf_root

from mmf.datasets.builders.sensetimeVAD.dataset_helper import *

from PIL import Image
import logging

class DetectionRecord(object):
    def __init__(self, row, root_=''):
        self._data = row
        self.root_ = root_
        self.path_idx = 0
        self.num_frame_idx = 1
        self.start_idx = 2
        self.end_idx = 3
        self.label_idx = 4
        self.rois_idx = 5
        self.frame_rate_idx = 6

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def start(self):
        return int(self._data[2])

    @property
    def end(self):
        return int(self._data[3])

    @property
    def label(self):
        return int(self._data[4])

    @property
    def rois(self):
        if len(self._data) < 6 or self._data[5] is None:
            return None
        else:
            return os.path.join(self.root_, self._data[5])

    @property
    def frame_rate(self):
        return 30

    @property
    def aux(self):
        try:
            return int(self._data[7]) + 1
        except:     # noqa
            if self._data[0].find('mta') >= 0:
                return int(2)
            return int(1)

    @property
    def raw_data(self):
        return [self.path, self.num_frames, self.start,
                self.end, self.label, self.rois, self.frame_rate]

class SensetimeVADDataset(BaseDataset):
    """Dataset for CLEVR. CLEVR is a reasoning task where given an image with some
    3D shapes you have to answer basic questions.

    Args:
        dataset_type (str): type of dataset, train|val|test
        config (DictConfig): Configuration Node representing all of the data necessary
                             to initialize CLEVR dataset class
        data_folder: Root folder in which all of the data will be present if passed
                     replaces default based on data_dir and data_folder in config.

    """

    def __init__(self, config, dataset_type, data_folder=None, *args, **kwargs):
        super().__init__(_CONSTANTS["dataset_key"], config, dataset_type)
        self.train_source_list = config.train_source
        self.val_source_list = config.val_source

        if dataset_type=='train':
            self.meta_file = config.train_source
            self.video_meta = self._parse_list(self.train_source_list)

        elif dataset_type == 'val' or dataset_type=='test':
            self.meta_file = config.val_source
            self.video_meta = self._parse_list(self.val_source_list)

        self.num_segments = config.num_segments
        self.new_length = config.new_length

        # process mot related
        self.scan_interval = config.scan_interval
        self.bbox_expand_ratio = config.bbox_expand_ratio
        self.filter_motor = config.filter_motor
        self.use_mannequin = config.use_mannequin
    
        self.ignore_squat = config.ignore_squat

        self.image_tmpl = config.image_tmpl
        self.merge_use = config.merge_data


    def _parse_list(self):
        video_list = []
        for meta_file_ in self.meta_file:
            for x in open(meta_file_):
                data = x.rstrip().split(';')
                num_frames = data[1]
                if int(data[2]) > int(data[3]):
                    logging.info('>>>> invalid record:{}'.format(data[0]))
                    continue
                elif int(num_frames) != int(data[3]) - int(data[2]) + 1:
                    data[1] = str(int(data[3]) - int(data[2]) + 1)
                    x = ';'.join(data) + '\n'

                tag_name = data[0] + '_' + data[2] + '_' + data[3]

                video_list.extend([
                    DetectionRecord(x.strip().split(';'))
                ])

            logging.debug('List [{}/{}] {} loaded.'.format(
                i + 1, len(self.meta_file), meta_file_)
            )

        return video_list

    def _parse_kmot_tracklet(self, path):
        """
        Parsing the KMOT results.
        Params:
            path: str. Path to kmot result.
        Return:
            tracklet_info: dict([list]):
                key: frame_index;
                value: tracklets of the correspoinding frame.
        """
        tracklet_info = defaultdict(list)
        lines = []
        if path is not None:
            if os.path.exists(path):
                with open(path, 'r') as fp:
                    lines = [line.strip().split(',') for line in fp]
                for item_ in lines:
                    if path.find('mta') >= 0:
                        frame_idx = int(item_[0])
                    else:
                        frame_idx = int(item_[0]) + 1
                    frame_content = item_[1:]
                    tracklet_info[frame_idx].append(frame_content)
        except:     # noqa
            raise Exception("Input error!", path)

        return tracklet_info

    def _sample_indices(self, record):

        start = record.start
        ave_dur = (record.num_frames - self.new_length + 1) // self.num_segments

        if ave_dur > 0:

            indices = start + np.multiply(list(range(self.num_segments)), ave_dur) + \
            randint(ave_dur, size=self.num_segments)

        elif record.num_frames < self.num_segments:
            indices = start + np.sort(randint(record.num_frames - self.new_length + 1,
                size=self.num_segments))
        else:

            indices = np.zeros((self.num_segments, )) + 1
            logging.info('{}, Video frames are null.'.format(record.path))

        return indices

    def _pick_mot_roi_by_indices(self,indices,tracklet_info, ref_size):

        multi_seg_tracklet_rst = []

        for seg_ind, frame_ind in enumerate(indices):

            p = int(frame_ind)

            for i in range(self.new_length):

                rois = load_and_filter_rois(
                    tracklet_info, p, self.scan_interval, self.bbox_expand_ratio,
                    self.filter_motor, self.use_mannequin, self.ignore_squat, ref_size)

                rois_num = len(rois)

                if rois_num > 0:
                    rois = np.array(rois)
                    ix = np.full((rois_num, 1), seg_ind)
                    rst_with_seg_idx = np.concatenate([ix, rois], axis=1)
                    multi_seg_tracklet_rst.extend(rst_with_seg_idx.tolist())
                if p < record.end:
                    p += 1

        return multi_seg_tracklet_rst

    def _get_img_file_temp(self,record,indices):

        image_tmpl = self.image_tmpl

        if record.path.find('mta') >= 0 and self.merge_use:
            self.merge_mta = True
        else:
            self.merge_mta = False

        if record.rois:
            if record.rois.find('mta') >= 0:
                image_tmpl = '{:08d}.jpeg'
                fake_flag = 1
            elif record.rois.find('mining') >= 0 and record.rois.find('tumble') < 0:
                image_tmpl = 'img_{:05d}.jpg'
                fake_flag = 0
            else:
                fake_flag = 0

        if record.path.find('faceunlock.tumbleData') >= 0:
            image_tmpl = '{:05d}.jpg'
            indices += 1

        return image_tmpl,indices

    def _load_image(self, filepath):

        img = [Image.open(filepath).convert('RGB')]

        return img


    def _get_gt_patch(self,indices,frame_path,mot_box):
        bbox_with_tag_rois = list()
        bbox_level_label = list()

        for k in range(len(indices)):

            frame_indice = indices[k]
            image_name = 'image_{:05d}.jpg.json'

            bbox_frame_dir = frame_path.replace('/frames/', '/bbox_tag/fighting/')

            if not os.path.exists(bbox_frame_dir):
                bbox_frame_dir = frame_path.replace('/frames/', '/bbox_tag/')

            image_file = os.path.join(bbox_frame_dir, image_name)

            shift = 0
            rst = None

            while shift < self.scan_interval:
                right_idx = indice + shift

                if os.path.isfile(image_file.format(right_idx)):
                    rst = image_file.format(right_idx)
                    break
                left_idx = indice - shift
                if os.path.isfile(image_file.format(left_idx)):
                    rst = image_file.format(left_idx)
                    break
                shift += 1

            if rst is None:
                bbox_tag_rois.append(mot_box[k])  # 没有对应标注，用预测的结果
                continue

            image_anno = open(rst, 'r').read()
            image_anno_rst = json.loads(image_anno)

            try:
                roi_rst = image_anno_rst['step_1']['result']
                if len(roi_rst) < 1:
                    bbox_tag_rois.append(mot_box[k])  # 没有对应标注，用预测的结果
                    continue
            except:    # noqa
                bbox_tag_rois.append(mot_box[k])  # 没有对应标注，用预测的结果
                continue

            logging.debug('roi number:{}'.format(len(roi_rst)))
            ## TODO: bbox id linking                           # noqa
            roi = [roi_rst[0]['x'], roi_rst[0]['y'],
                   roi_rst[0]['x'] + roi_rst[0]['width'],
                   roi_rst[0]['y'] + roi_rst[0]['height']]
            bbox_tag_rois.append(roi)

        bbox_tag_rois = np.array(bbox_tag_rois)

        assert bbox_tag_rois.shape[0] == self.num_segments and bbox_tag_rois.shape[1] == 4
        
        return bbox_tag_rois



    def __len__(self):
        return len(self.video_meta)


    def __getitem__(self, idx):

        record = self.video_meta[idx]
        sampled_frame_indices = self._sample_indices 

        video_path = record.path 
        video_start = record.start
        video_en = record.end
        video_level_label = record.label


        image_tmpl,indices = self._get_img_file_temp(record,sampled_frame_indices)

        ref_img_filename = reading_with_exception_handling(record, image_tmpl, int(indices[0]))
        temp_img_ = self._load_image(ref_img_filename)[0]
        ref_size = temp_img_.size

        tracklet_info = self._parse_kmot_tracklet(record.rois)
        filtered_boxes_with_track_by_indice = self._pick_mot_roi_by_indices(sampled_frame_indices,tracklet_info,ref_size)


        gt_boxes_info = self._get_gt_patch(indices,video_path,filtered_boxes_with_track_by_indice)

        input_box = self._generate_input()



        current_sample = Sample()
        current_sample.targets = 
        current_sample.human_box = 

        return current_sample
