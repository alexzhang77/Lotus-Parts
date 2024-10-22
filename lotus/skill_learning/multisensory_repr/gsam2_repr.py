import torch
from torchvision import transforms
import numpy as np
import cv2
import argparse
import h5py
import os
from functools import partial

import sys
sys.path.append('Grounded_SAM_2')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from einops import rearrange
from easydict import EasyDict
import torch.backends.cudnn as cudnn

from sklearn.decomposition import PCA
from models.model_utils import safe_cuda

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import requests
from pathlib import Path
from supervision.draw.color import ColorPalette
from PIL import Image


URL = "http://127.0.0.1:8000/get_arr_embeddings"

Dataset_Name_List = [
    "../datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo",
    "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo",
]

def rescale_feature_map(img_tensor, target_h, target_w, convert_to_numpy=True):
    img_tensor = torch.nn.functional.interpolate(img_tensor, (target_h, target_w))
    if convert_to_numpy:
        return img_tensor.cpu().numpy()
    else:
        return img_tensor

def process_images(imgs, prompt):
    # imgs should be a batch of images, shape (batch_size, height, width, channels)
    sizes = [448, 224]
    max_size = max(sizes) // 14
    batch_size = len(imgs)

    all_features = []
    for size in sizes:
        imgs_resized = [cv2.resize(img, (size, size)) for img in imgs]
        prompt = prompt[:-5]
        features = [np.array(requests.post(URL, json={"img_arr": img.tolist(), "prompt": prompt}).json()['embeddings']) for img in imgs_resized]
        new_feats = np.array(features).squeeze()
        print(new_feats.shape)
        new_feats = torch.nn.functional.interpolate(torch.from_numpy(new_feats), (max_size, max_size), mode="bilinear", align_corners=True, antialias=True)
        new_feats = rearrange(new_feats, 'b c h w -> b h w c')
        all_features.append(new_feats)

    all_features = torch.mean(torch.stack(all_features), dim=0)
    return all_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-name',
        type=str,
        default="debug",
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=768*2,
    )
    parser.add_argument(
        '--modality-str',
        type=str,
        default="gsam2_agentview_eye_in_hand",
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=100
    )
    args = parser.parse_args()
    modality_str = args.modality_str
    feature_dim = args.feature_dim

    for dataset_name in Dataset_Name_List:
        dataset_hdf5_file = dataset_name + ".hdf5"
        f = h5py.File(dataset_hdf5_file, "r")
        demo_num = len(f['data'].keys())

        dataset_name_parts = dataset_name.split("/")
        part_2 = dataset_name_parts[-2]
        part_1 = dataset_name_parts[-1]
        embedding_name = f"results/{args.exp_name}/repr/{part_2}/{part_1}/embedding_{modality_str}_{feature_dim}.hdf5"
        os.makedirs(os.path.dirname(embedding_name), exist_ok=True)
        print("Saving embedding to", embedding_name)
        embedding_file = h5py.File(embedding_name, "w")
        grp = embedding_file.create_group("data")

        # for each demo, generate agent view and eye in hand feature embeddings as numpy arrays of shape (batch size, 1, channels * 2 (6?)) and save in corresponding hdf5
        for i in range(demo_num):
            agentview_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/agentview_rgb"][()].transpose(0, 3, 1, 2))).float()
            eye_in_hand_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2))).float()
            joint_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/joint_states"][()])).float()
            gripper_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/gripper_states"][()])).float()
            ee_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/ee_states"][()])).float()
            proprio_states = torch.cat([joint_states, gripper_states, ee_states], dim=1).float()
            proprio = safe_cuda(proprio_states)
            agentview_features = []
            eye_in_hand_features = []

            for j in range(0, len(agentview_images), args.batch_size):
                batch_images = agentview_images[j:j + args.batch_size].permute(0, 2, 3, 1).cpu().numpy()
                resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in batch_images]
                features = process_images(resized_images, part_1.replace("_", " "))
                agentview_features_batch = rescale_feature_map(torch.as_tensor(features).permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze()  # (B, 768)
                if agentview_features_batch.dim() == 1:
                    agentview_features_batch = agentview_features_batch.unsqueeze(0)
                agentview_features.append(agentview_features_batch)

            for j in range(0, len(eye_in_hand_images), args.batch_size):
                batch_images = eye_in_hand_images[j:j + args.batch_size].permute(0, 2, 3, 1).cpu().numpy()
                resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in batch_images]
                features = process_images(resized_images, part_1.replace("_", " "))
                eye_in_hand_features_batch = rescale_feature_map(torch.as_tensor(features).permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze()  # (B, 768)
                if eye_in_hand_features_batch.dim() == 1:
                    eye_in_hand_features_batch = eye_in_hand_features_batch.unsqueeze(0)
                eye_in_hand_features.append(eye_in_hand_features_batch)

            agentview_features = torch.cat(agentview_features, dim=0)
            eye_in_hand_features = torch.cat(eye_in_hand_features, dim=0)
            embeddings = torch.cat([agentview_features, eye_in_hand_features], dim=1).cpu().unsqueeze(1).numpy().astype('float32')
            if np.isnan(embeddings).any():
                print("NAN")

            demo_data_grp = grp.create_group(f"demo_{i}")
            demo_data_grp.create_dataset("embedding", data=embeddings)

        embedding_file.close()
        f.close()