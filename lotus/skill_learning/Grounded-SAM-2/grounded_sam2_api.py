import argparse
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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from io import BytesIO


from fastapi import FastAPI, UploadFile, File, Form

# Create an instance of FastAPI
app = FastAPI()



'''''''''
Here we will be loading up the models first
'''

# main variables we will be using 
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE TYPE: ", DEVICE)

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
print("Building the model")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
print("Getting the predictor")
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
print("Getting the pretrained model")
processor = AutoProcessor.from_pretrained(model_id)
print("Getting the grounding model")
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)




@app.post("/get_embeddings")
async def get_embeddings(img: UploadFile = File(...), prompt: str = Form(...)):
    
    print("Called get_embeddings")


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = prompt

    print("Prompt: ", prompt)

    # prepare the image 
    contents = await img.read()
    image = Image.open(BytesIO(contents))

    image.save("/home/davin123/SAM-Server/Grounded-SAM-2/request_image.png", format="PNG")

    print("Predicting image embeddings")
    sam2_predictor.set_image(np.array(image.convert("RGB")))


    print("Output from dino model")


    # Get output from dino
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    
    print("Outputs: ", outputs)


    # this is to filter out the responses in which the system recieved 
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

    print("BOXES: ", input_boxes)

    print("Get sparse and dense embeddings")


    sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )


    sparse_embeddings, dense_embeddings = sam2_predictor.predict_sparse_and_dense_embeddings(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    print("Get image embeddings")

    image_embeddings = sam2_predictor.get_image_embedding()

    print("IMAGE EMBED: ", image_embeddings)
    print("IMAGE EMBED SIZE: ", image_embeddings.size())

    return {"sparse_embeddings": sparse_embeddings.tolist(), "dense_embeddings": dense_embeddings.tolist(), "image_embeddings": image_embeddings.tolist()}
