import torch
from torchvision import transforms
import numpy as np
import cv2
import argparse
import h5py
import os
from functools import partial

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
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from Gounded-SAM-2.sam2.build_sam import build_sam2
from Gounded-SAM-2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

"""
Hyper parameters
"""
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "car. tire."
# need to replace IMG_PATH with PIL img from hdf5 file
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_hf_model_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image = Image.open(img_path)

sam2_predictor.set_image(np.array(image.convert("RGB")))

inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

"""
Results is a list of dict with the following structure:
[
    {
        'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
        'labels': ['car', 'tire', 'tire', 'tire'], 
        'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                        [1392.4701,  554.4064, 1628.6133,  777.5872],
                        [ 436.1182,  621.8940,  676.5255,  851.6897],
                        [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
    }
]
"""

# get the box prompt for SAM 2
input_boxes = results[0]["boxes"].cpu().numpy()

mask_input, unnorm_coords, labels, unnorm_box = sam2_predictor._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords
        )


def rescale_feature_map(img_tensor, target_h, target_w, convert_to_numpy=True):
    img_tensor = torch.nn.functional.interpolate(img_tensor, (target_h, target_w))
    if convert_to_numpy:
        return img_tensor.cpu().numpy()
    else:
        return img_tensor

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
                features = dinov2.process_images(resized_images)
                agentview_features_batch = rescale_feature_map(torch.as_tensor(features).permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze()  # (B, 768)
                if agentview_features_batch.dim() == 1:
                    agentview_features_batch = agentview_features_batch.unsqueeze(0)
                agentview_features.append(agentview_features_batch)

            for j in range(0, len(eye_in_hand_images), args.batch_size):
                batch_images = eye_in_hand_images[j:j + args.batch_size].permute(0, 2, 3, 1).cpu().numpy()
                resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in batch_images]
                features = dinov2.process_images(resized_images)
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