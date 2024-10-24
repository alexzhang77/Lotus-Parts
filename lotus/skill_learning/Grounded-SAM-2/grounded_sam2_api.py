import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from pydantic import BaseModel
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from einops import rearrange

from io import BytesIO


from fastapi import FastAPI, UploadFile, File, Form

# Create an instance of FastAPI
app = FastAPI()

class ArrayInput(BaseModel):
    img_arr: list
    prompt: str

'''''''''
Here we will be loading up the models first
'''

# main variables we will be using 
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "./request_image.png"

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

    image.save(SAVE_FILE, format="PNG")

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
    print("Sparse EMBED SIZE: ", sparse_embeddings.size())
    print("Dense EMBED SIZE: ", dense_embeddings.size())

    return {"sparse_embeddings": sparse_embeddings.tolist(), "dense_embeddings": dense_embeddings.tolist(), "image_embeddings": image_embeddings.tolist()}

@app.post("/get_arr_embeddings")
async def get_arr_embeddings(img_json: ArrayInput):


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = img_json.prompt

    # prepare the image 
    np_img = np.array(img_json.img_arr)
    image = Image.fromarray(np.uint8(np_img)).convert('RGB')
    image.save(SAVE_FILE, format="PNG")
    np_img = np.uint8(np_img)
    sam2_predictor.set_image(np_img)
    # Get output from dino
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)


    # this is to filter out the responses in which the system recieved 
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )


    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()

    if (input_boxes.shape != (0,4)):

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

        image_embeddings = sam2_predictor.get_image_embedding()
        print(f"Image shape: {image_embeddings.shape}")
        print(f"Dense shape: {dense_embeddings.shape}")
        f_embeddings = torch.mean(torch.cat((image_embeddings,dense_embeddings),0), 0).unsqueeze(0)
        print(f"Final shape: {f_embeddings.shape}")
    
    else:
        f_embeddings = sam2_predictor.get_image_embedding()
        image_embeddings = f_embeddings
        sparse_embeddings = np.zeros(0)
        dense_embeddings = np.zeros(0)
    return {"embeddings": f_embeddings.tolist(), "sparse_ embeddings": sparse_embeddings.tolist(), "dense_embeddings": dense_embeddings.tolist(), "image_embeddings": image_embeddings.tolist()}