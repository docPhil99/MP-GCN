import pickle
import json
import logging
import numpy as np
import os
from torch.utils.data import Dataset
from .utils import graph_processing, multi_input


class camD_Feeder(Dataset):
    def __init__(self, phase, graph, root_folder, inputs, debug, obj=False, max_frame=80, object_folder='', processing='default', person_id=[0], input_dims=2, **kwargs):
        self.phase = phase
        self.inputs = inputs
        self.processing = processing
        self.obj = obj
        self.debug = debug
        
        self.graph = graph.graph
        self.conn = graph.connect_joint
        self.center = graph.center
        self.num_node = graph.num_node
        self.num_person = graph.num_person
        
        self.input_dims = input_dims # expected to be 3 (x,y,conf)
        self.max_frame = max_frame
        self.M = len(person_id)
        self.datashape = self.get_datashape()

        data_path = os.path.join(root_folder, phase+'_data.npy')
        label_path = os.path.join(root_folder, phase+'_label.pkl')
        object_json_path = os.path.join(object_folder, phase+'_object_data.json')

        try:
            logging.info('Loading {} pose data from {}'.format(phase, data_path))
            self.data = np.load(data_path)
            # N, T, M, V, C -> N, C, T, V, M
            self.data = self.data.transpose(0, 4, 1, 3, 2)
            
            with open(label_path, 'rb') as f:
                self.label = pickle.load(f)
            
            if obj:
                logging.info('Loading {} object data from '.format(phase) + object_json_path)
                self.object_data, self.object_names = self.load_obj_data(object_json_path, phase)
                # (N, T, v, C) -> (N, C, T, v)
                self.object_data = self.object_data.transpose(0, 3, 1, 2)
                
                # (N, C, T, v) -> (N, C, T, v, M)
                self.object_data = np.expand_dims(self.object_data, axis=-1)
                self.object_data = np.tile(self.object_data, (1, 1, 1, 1, self.M))
                
                # (N, C, T, V, M) -> (N, C, T, V+v, M)
                self.data = np.concatenate((self.data, self.object_data), axis = 3)
            else:
                # If no object integration requested, set object_names to None
                self.object_names = None
            
            self.data = self.data[:, :self.input_dims, :, :, :]

        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(
                data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if self.debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            if self.object_names is not None:
                self.object_names = self.object_names[:300]
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # (C, T, V, M)
        pose_data = np.array(self.data[idx])
        label, name = self.label[idx]
        
        pose_data = graph_processing(pose_data, self.graph, self.processing)
        data_new = multi_input(pose_data, self.conn, self.inputs, self.center)
        
        try:
            assert list(data_new.shape) == self.datashape
        except AssertionError:
            logging.info('data_new.shape: {}'.format(data_new.shape))
            raise ValueError()
        
        # object name to return
        obj_name = None
        if self.obj and self.object_names is not None:
            # safe guard: object_names length may be shorter; default to "unknown"
            if idx < len(self.object_names):
                obj_name = self.object_names[idx]
            else:
                obj_name = "unknown"
        
        return data_new, label, name, obj_name
    
    def get_datashape(self):
        I = len(self.inputs) if self.inputs.isupper() else 1
        C = self.input_dims if self.inputs in [
            'joint', 'joint-motion', 'bone', 'bone-motion'] else self.input_dims*2
        T = self.max_frame
        V = self.num_node
        M = self.M // self.num_person
        return [I, C, T, V, M]
    

    def load_obj_data(self, object_json_path, phase):
        """We load the object data here seperately."""

        logging.info('Loading {} object data from {}'.format(phase, object_json_path))
        if not os.path.exists(object_json_path):
            logging.error(f"Expected object JSON at {object_json_path} but not found.")
            raise ValueError("Missing object JSON file")

        # object_json is a list (len N) where each element is a dict:
        # e.g. {"obj_name": [[x1,y1,confidence], [x2,y2,confidence], ...]}
        with open(object_json_path, 'r') as f:
            res_obj = json.load(f)

        # Build numpy array with shape (N, T, v=1, C=3)
        obj_list = []
        obj_names = []
        N_pose = self.data.shape[0]  # number of samples from pose file

        for i, obj_item in enumerate(res_obj):
            # obj_item expected to be a dict with single key: name -> list of [x,y,confidence] per frame
            if obj_item is None:
                # keep placeholder
                obj_list.append(None)
                obj_names.append("unknown")
                continue

            if isinstance(obj_item, dict) and len(obj_item) >= 1:
                # get first key (object name) and coordinate list
                k = next(iter(obj_item.keys()))
                coords = obj_item[k]  # list of [x, y, confidence] per frame
                coords = np.array(coords, dtype=np.float32)  # (T_obj, 3)
                obj_list.append(coords)
                obj_names.append(k)
            else:
                # unexpected entry, push NaNs
                obj_list.append(None)
                obj_names.append("unknown")

        # Convert list entries to shape (N, T, 1, 3)
        # Pose T:
        _, _, T_pose, V_pose, M_pose = self.data.shape  # after transpose
        # original shape: (N, C, T, V, M)
        obj_np = np.zeros((N_pose, T_pose, 1, 3), dtype=np.float32)

        for i, coords in enumerate(obj_list):
            if coords is None:
                obj_np[i, :, 0, :] = 0.0
            else:
                # coords might be shorter/longer than T_pose — handle by trunc/pad with nan
                t_len = coords.shape[0]
                if t_len >= T_pose:
                    obj_np[i, :, 0, :] = coords[:T_pose, :]
                else:
                    obj_np[i, :t_len, 0, :] = coords
                    obj_np[i, t_len:, 0, :] = 0.0

        return obj_np, obj_names