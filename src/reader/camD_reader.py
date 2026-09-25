import os
import json
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold  # Added KFold import

ACTIVITIES = ['B1', 'F1', 'B2', 'F2', 'B3',
              'F3', 'B4', 'F4', 'B5', 'F5']

class CamD_Reader():
    def __init__(self, dataset_root_folder, out_folder, n_splits=10, seed=42, **kwargs):
        self.max_channel = 3
        self.max_frame = 100
        self.max_joint = 17
        self.max_person = 6
        self.min_frame_id = 10 # skip first n frames
        self.last_n_frames = 4 # skip last n frames

        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder
        self.n_splits = n_splits

        # Pose directory & file listing
        self.pose_dir = os.path.join(dataset_root_folder, 'poses')
        self.video_files = np.array(os.listdir(self.pose_dir))

        # Setup K-Fold Cross Validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        
        # Store index arrays for each fold: list of (train_idx, val_idx)
        self.folds = list(kf.split(self.video_files))

        # Create label-to-idx map
        self.class2idx = {name: i for i, name in enumerate(ACTIVITIES)}


    def read_pose_and_object(self, pose_path, obj_path):
        T, M, V, C = self.max_frame, self.max_person, self.max_joint, self.max_channel

        # --- Initialize pose data array ---
        skeleton_data = np.zeros((T, M, V, C), dtype=np.float32)

        # --- Load Pose JSON ---
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        # Fill skeleton_data
        frame_cnt = 0
        reduced_list = self.downsample_frames(pose_data)

        for frame_data in pose_data.get("frames", {}):
            t = frame_data["frame_index"]

            if t not in reduced_list:
                continue
            
            for m, person in enumerate(frame_data["poses"][:M]):
                kpts = np.array(person["keypoints"], dtype=np.float32)
                conf = np.array(person["confidence"], dtype=np.float32)
                joint = np.concatenate([kpts, conf[:, None]], axis=1)
                v = min(V, joint.shape[0])
                skeleton_data[frame_cnt, m, :v, :] = joint[:v]

            frame_cnt += 1

        # --- Load Object JSON ---
        with open(obj_path, 'r') as f:
            obj_data = json.load(f)

        obj_name = None
        obj_coords = []
        
        for frame_data in obj_data.get("frames", []):
            t = frame_data["frame_index"]
            if t not in reduced_list:
                continue

            objects = frame_data.get("objects", [])
            if not objects:
                obj_coords.append([0.0, 0.0, 0.0])
                continue

            obj = max(objects, key=lambda o: o.get("confidence", 0))
            if obj_name is None:
                obj_name = obj["object_name"]
            cx, cy = obj["center"]
            conf = obj["confidence"]
            obj_coords.append([float(cx), float(cy), float(conf)])

        object_info = {obj_name or "unknown": obj_coords}

        return skeleton_data, object_info


    def gendata(self, sample_indices, fold_out_dir, phase):
        res_skeleton = []
        res_obj = []
        group_label = []
        
        videos = self.video_files[sample_indices].tolist()
        iterizer = tqdm(videos, desc=f"{phase.capitalize()}", dynamic_ncols=True)
        
        for filename in iterizer:
            video_id = filename.split('.')[0].split('_')[0]

            # Skip random walking files
            if video_id[:-5] not in ['RED', 'YELLOW', 'BLACK', 'GREEN', 'BLUE', 'WHITE']:
                continue

            joint_path = os.path.join(self.dataset_root_folder, 'poses', filename)
            object_path = os.path.join(self.dataset_root_folder, 'objects', f'{video_id}_left_objects.json')
            
            group_label.append([self.class2idx[video_id[-5:-3]], video_id])
                
            joint_data, object_data = self.read_pose_and_object(joint_path, object_path)
            res_skeleton.append(joint_data)
            res_obj.append(object_data)
                
        # Create fold output subfolder
        os.makedirs(fold_out_dir, exist_ok=True)

        # Save outputs per phase into the fold directory
        with open(os.path.join(fold_out_dir, f'{phase}_label.pkl'), 'wb') as f:
            pickle.dump(group_label, f)
        
        res_skeleton = np.array(res_skeleton)
        np.save(os.path.join(fold_out_dir, f'{phase}_data.npy'), res_skeleton)
        
        with open(os.path.join(fold_out_dir, f'{phase}_object_data.json'), "w") as f:
            json.dump(res_obj, f)


    def start(self):
        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            logging.info(f'--- Processing Fold {fold_idx + 1}/{self.n_splits} ---')
            fold_dir = os.path.join(self.out_folder, f'fold_{fold_idx}')
            
            # Generate Train and Eval sets for this fold
            self.gendata(train_idx, fold_dir, phase='train')
            self.gendata(val_idx, fold_dir, phase='eval')


    def downsample_frames(self, pose_data, random_idx=False):
        frames_dict = pose_data.get("frames", {})
        len_frames = len(frames_dict)
        
        T = len_frames - self.min_frame_id - self.last_n_frames
        target_frame_count = self.max_frame
        
        if target_frame_count >= T:
            return list(range(self.min_frame_id, T))
        
        if not random_idx:
            indices = [self.min_frame_id + int(i * T / target_frame_count) for i in range(target_frame_count)]
        else:
            indices = random.sample(range(self.min_frame_id, T), self.max_frame)
            indices.sort()

        return indices