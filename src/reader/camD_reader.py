import os
import json
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm

ACTIVITIES = ['B1', 'F1', 'B2', 'F2', 'B3',
              'F3', 'B4', 'F4', 'B5', 'F5']

class CamD_Reader():
    def __init__(self, dataset_root_folder, out_folder, split_ratio=0.7, **kwargs):
        self.max_channel = 3
        self.max_frame = 100
        self.max_joint = 17
        self.max_person = 6
        self.min_frame_id = 10 # skip first n frames
        self.last_n_frames = 4 # skip last n frames

        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder

        # Divide train and eval samples
        self.pose_dir = os.path.join(dataset_root_folder,'poses')
        smp_idx = list(range(len(os.listdir(self.pose_dir))))
        random.shuffle(smp_idx)
        split_idx = int(len(smp_idx) * split_ratio)
        self.training_samples = smp_idx[:split_idx]
        self.eval_samples = smp_idx[split_idx:]

        # Create label-to-idx map
        self.class2idx = {name: i for i, name in enumerate(ACTIVITIES)}


    def read_pose_and_object(self, pose_path, obj_path):
        """
        Reads pose and object JSON files and returns:
        - skeleton_data: np.array of shape (T, M, V, C)
        where C = [x, y, confidence]
        - object_info: dict {"obj_name": [[x,y,confidence], [x,y,confidence], ...]}

        Args:
            pose_path (str): Path to pose JSON files.
            obj_path (str): Directory containingPath to object JSON files.
            T (int): Total number of frames (optional fixed length).
            V (int): Number of joints per person.
            M (int): Number of persons (actors).
            C (int): Channels per joint (x, y, conf).

        Returns:
            skeleton_data (np.ndarray): shape (T, M, V, C)
            object_info (dict): {"obj_name": [[x, y], ...]}
        """

        T, M, V, C = self.max_frame, self.max_person, self.max_joint, self.max_channel

        # --- Initialize pose data array ---
        skeleton_data = np.zeros((T, M, V, C), dtype=np.float32)

        # --- Load Pose JSON ---
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        # Fill skeleton_data
        frame_cnt = 0 #used to populate the skeleton array (index)
        
        reduced_list = self.downsample_frames(pose_data)

        for frame_data in pose_data.get("frames", {}):
            
            t = frame_data["frame_index"]

            # Reduce frame rate
            if t not in reduced_list:
                continue
            
            for m, person in enumerate(frame_data["poses"][:M]):
                kpts = np.array(person["keypoints"], dtype=np.float32)
                conf = np.array(person["confidence"], dtype=np.float32)
                # Shape: (V, 2) → stack with conf to get (V, 3)
                joint = np.concatenate([kpts, conf[:, None]], axis=1)
                v = min(V, joint.shape[0])
                skeleton_data[frame_cnt, m, :v, :] = joint[:v]

            frame_cnt += 1 # increase count till we get to last frame

        # --- Load Object JSON ---
        with open(obj_path, 'r') as f:
            obj_data = json.load(f)

        # Assuming one object of interest per video
        obj_name = None
        obj_coords = []
        
        for frame_data in obj_data.get("frames", []):

            t = frame_data["frame_index"]
            # Reduce frame rate
            if t not in reduced_list:
                continue

            objects = frame_data.get("objects", [])
            if not objects:
                obj_coords.append([0.0, 0.0, 0.0])  # no object in frame
                continue
            # Pick first object (or highest confidence)
            obj = max(objects, key=lambda o: o.get("confidence", 0))
            if obj_name is None:
                obj_name = obj["object_name"]
            cx, cy = obj["center"]
            conf = obj["confidence"]
            obj_coords.append([float(cx), float(cy), float(conf)])

        object_info = {obj_name or "unknown": obj_coords}

        return skeleton_data, object_info


    def gendata(self, phase):

        res_skeleton = []
        res_obj = []
        group_label = []
        video_list = np.array(os.listdir(self.pose_dir)) # use the directory for pose
        videos = video_list[self.training_samples].tolist() if phase == 'train' else video_list[self.eval_samples].tolist()
        
        iterizer = tqdm(videos, dynamic_ncols=True)
        for filename in iterizer:

            video_id = filename.split('.')[0].split('_')[0] # loading from a .json file

            # Skip the random walking files
            if video_id[:-5] not in ['RED', 'YELLOW', 'BLACK', 'GREEN', 'BLUE', 'WHITE']:
                continue

            # path to joints and object files
            joint_path = os.path.join(
                self.dataset_root_folder, 'poses', filename)
            object_path = os.path.join(
                self.dataset_root_folder, 'objects', f'{video_id}_left_objects.json')
            
            # save group name for each video sample
            group_label.append([self.class2idx[video_id[-5:-3]], video_id])
                
                
            # Get joint/+object information
            joint_data, object_data = self.read_pose_and_object(joint_path, object_path)
            res_skeleton.append(joint_data)
            res_obj.append(object_data)
                
        # Save label
        os.makedirs(self.out_folder, exist_ok=True)
        with open(os.path.join(self.out_folder, phase + '_label.pkl'), 'wb') as f:
            pickle.dump(group_label, f)
        
        # Save pose data
        res_skeleton = np.array(res_skeleton)
        np.save(os.path.join(self.out_folder, phase + '_data.npy'), res_skeleton)
        
        # Save obj data
        with open(os.path.join(self.out_folder, phase + '_object_data.json'), "w") as f:
            json.dump(res_obj, f)
        
    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)


    def downsample_frames(self, pose_data, random_idx=False):
        """
        Downsample a list of video frames to a specified target length.
        
        Args:
            pose_data (dict): dictionary of pose information.
            target_frame_count (int): The desired number of frames after downsampling.
            
        Returns:
            list: A uniformly downsampled list of frames.
        """

        frames_dict = pose_data.get("frames", {})

        len_frames = len(frames_dict) # length of video frames
        
        T = len_frames - self.min_frame_id - self.last_n_frames

        target_frame_count = self.max_frame
        
        # If the video is already short enough, return as is or pad if needed
        if target_frame_count >= T:
            return list(range(self.min_frame_id,T))
        
        if not random_idx:
        
            # Compute indices for uniform sampling
            indices = [self.min_frame_id + int(i * T / target_frame_count) for i in range(target_frame_count)]
        else:
            indices = random.sample(range(self.min_frame_id, T), self.max_frame)
            indices.sort()
        
        # Ensure last index doesn't exceed bounds
        # indices[-1] = min(indices[-1], T - 1)

        return indices