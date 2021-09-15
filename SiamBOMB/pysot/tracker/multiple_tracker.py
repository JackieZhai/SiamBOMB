# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 

import time
import numpy as np
from copy import deepcopy
from collections import OrderedDict


class MultiTracker:
    def __init__(self, tracker, cfg):
        self.tracker = tracker
        self.cfg = cfg

        self.initialized_ids = []
        self.trackers = OrderedDict()
    
    def _split_info(self, info):
        info_split = OrderedDict()
        init_other = OrderedDict()  # Init other contains init info for all other objects
        for obj_id in info['init_object_ids']:
            info_split[obj_id] = dict()
            init_other[obj_id] = dict()
            info_split[obj_id]['object_ids'] = [obj_id]
            info_split[obj_id]['sequence_object_ids'] = info['sequence_object_ids']
            if 'init_bbox' in info:
                info_split[obj_id]['init_bbox'] = info['init_bbox'][obj_id]
                init_other[obj_id]['init_bbox'] = info['init_bbox'][obj_id]
            if 'init_mask' in info:
                info_split[obj_id]['init_mask'] = (info['init_mask'] == int(obj_id)).astype(np.uint8)
                init_other[obj_id]['init_mask'] = info_split[obj_id]['init_mask']
        for obj_info in info_split.values():
            obj_info['init_other'] = init_other
        return info_split

    def _set_defaults(self, tracker_out: dict, defaults=None):
        defaults = {} if defaults is None else defaults

        for key, val in defaults.items():
            if tracker_out.get(key) is None:
                tracker_out[key] = val

        return tracker_out
    
    def default_merge(self, out_all):
        out_merged = OrderedDict()

        out_first = list(out_all.values())[0]
        out_types = out_first.keys()

        # Merge segmentation mask
        if 'segmentation' in out_types and out_first['segmentation'] is not None:
            # Stack all masks
            # If a tracker outputs soft segmentation mask, use that. Else use the binary segmentation
            segmentation_maps = [out.get('segmentation_soft', out['segmentation']) for out in out_all.values()]
            segmentation_maps = np.stack(segmentation_maps)

            obj_ids = np.array([0, *map(int, out_all.keys())], dtype=np.uint8)
            segm_threshold = getattr(self.cfg, 'TRACK.MASK_THERSHOLD', 0.5)
            merged_segmentation = obj_ids[np.where(segmentation_maps.max(axis=0) > segm_threshold,
                                                   segmentation_maps.argmax(axis=0) + 1, 0)]

            out_merged['segmentation'] = merged_segmentation

        # Merge other fields
        for key in out_types:
            if key == 'segmentation':
                pass
            else:
                out_merged[key] = {obj_id: out[key] for obj_id, out in out_all.items()}

        return out_merged
    
    def siammask_merge(self, out_all):
        out_merged = OrderedDict()

        out_first = list(out_all.values())[0]
        out_types = out_first.keys()

        # Merge segmentation mask
        if 'mask' in out_types and out_first['mask'] is not None:
            # Stack all masks
            segmentation_maps = [out['mask'] for out in out_all.values()]
            segmentation_list = []
            for seg in segmentation_maps:
                if seg is not None:
                    segmentation_list.append(seg)
            segmentation_maps = np.stack(segmentation_list)

            obj_ids = np.array([0, *map(int, out_all.keys())], dtype=np.uint8)
            segm_threshold = getattr(self.cfg, 'TRACK.MASK_THERSHOLD', 0.5)
            merged_segmentation = obj_ids[np.where(segmentation_maps.max(axis=0) > segm_threshold,
                                                   segmentation_maps.argmax(axis=0) + 1, 0)]

            out_merged['segmentation'] = merged_segmentation

        # Merge other fields
        for key in out_types:
            if key == 'mask':
                pass
            elif key == 'bbox':
                out_merged['target_bbox'] = {obj_id: out[key] for obj_id, out in out_all.items()}
            elif key == 'polygon':
                out_merged['target_polygon'] = {}
                for obj_id, out in out_all.items():
                    if key in out.keys():
                        out_merged['target_polygon'][obj_id] = out[key]
            elif key == 'time':
                out_merged['time'] = {obj_id: out[key] for obj_id, out in out_all.items()}

        return out_merged

    def merge_outputs(self, out_all):
        # out_merged = self.default_merge(out_all)
        out_merged = self.siammask_merge(out_all)

        return out_merged

    def initialize(self, image, info: dict) -> dict:
        self.initialized_ids = []
        self.trackers = OrderedDict()

        if len(info['init_object_ids']) == 0:
            return None

        object_ids = info['object_ids']

        init_info_split = self._split_info(info)
        self.trackers = OrderedDict({obj_id: deepcopy(self.tracker) for obj_id in object_ids})

        out_all = OrderedDict()
        # Run individual trackers for each object
        for obj_id in info['init_object_ids']:
            start_time = time.time()
            out = self.trackers[obj_id].init(image, init_info_split[obj_id]['init_bbox'])
            if out is None:
                out = {}

            init_default = {'bbox': init_info_split[obj_id].get('init_bbox'),
                            'time': time.time() - start_time,
                            'mask': init_info_split[obj_id].get('init_mask')}

            out = self._set_defaults(out, init_default)
            out_all[obj_id] = out

        self.initialized_ids = info['init_object_ids'].copy()

        # Merge results
        out_merged = self.merge_outputs(out_all)

        return out_merged

    def track(self, image, info: dict = None) -> dict:
        if info is None:
            info = {}

        prev_output = info.get('previous_output', OrderedDict())

        if info.get('init_object_ids', False) and len(info['init_object_ids']) > 0:
            init_info_split = self._split_info(info)
            # for obj_init_info in init_info_split.values():
            #     obj_init_info['previous_output'] = prev_output
            # info['init_other'] = list(init_info_split.values())[0]['init_other']
        if info.get('sequence_object_ids', False) and len(info['sequence_object_ids']) > 0:
            for obj_id in info['sequence_object_ids']:
                if obj_id not in self.initialized_ids:
                    self.initialized_ids.append(obj_id)
                
        out_all = OrderedDict()
        temp_initialized_ids = deepcopy(self.initialized_ids)
        for obj_id in temp_initialized_ids:
            start_time = time.time()

            if info.get('sequence_object_ids', False) and \
                obj_id in info['sequence_object_ids']:
                if obj_id in info['init_bbox']:  # Modify
                    new_bbox = info['init_bbox'][obj_id]
                    self.trackers[obj_id] = deepcopy(self.tracker)
                    out = self.trackers[obj_id].init(image, new_bbox)
                    if out is None:
                        out = {}
                    default = {'bbox': new_bbox,
                               'time': time.time() - start_time,
                               'mask': None}
                else:  # Delete
                    self.trackers.pop(obj_id)
                    self.initialized_ids.remove(obj_id)
                    continue
            else:
                out = self.trackers[obj_id].track(image)
                default = {'time': time.time() - start_time}

            out = self._set_defaults(out, default)
            out_all[obj_id] = out

        # Initialize new
        if info.get('init_object_ids', False) and len(info['init_object_ids']) > 0:
            for obj_id in info['init_object_ids']:
                if not obj_id in self.trackers:
                    self.trackers[obj_id] = deepcopy(self.tracker)

                start_time = time.time()
                out = self.trackers[obj_id].init(image, init_info_split[obj_id]['init_bbox'])
                if out is None:
                    out = {}

                init_default = {'bbox': init_info_split[obj_id].get('init_bbox'),
                                'time': time.time() - start_time,
                                'mask': init_info_split[obj_id].get('init_mask')}

                out = self._set_defaults(out, init_default)
                out_all[obj_id] = out

            self.initialized_ids.extend(info['init_object_ids'])

        # Merge results
        out_merged = self.merge_outputs(out_all)

        return out_merged