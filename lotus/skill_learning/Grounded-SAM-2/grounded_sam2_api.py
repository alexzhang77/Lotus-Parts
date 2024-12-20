import torch
import numpy as np
from pydantic import BaseModel
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from fastapi import FastAPI

# required for api content
import json

# Create an instance of FastAPI
app = FastAPI()

class ArrayInput(BaseModel):
    img_arr: list
    prompt: str

# class ArrayInput(BaseModel):
#     prompt: str

'''''''''
Here we will be loading up the models first
'''

# main variables we will be using 
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "./request_image.png"

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

# set it to batching mode
sam2_predictor._is_batch = True

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)



@app.post("/get_embeddings")
async def get_arr_embeddings(img_json: ArrayInput):


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = img_json.prompt

    # prepare the image 
    np_img = np.array(img_json.img_arr)
    image = Image.fromarray(np.uint8(np_img)).convert('RGB')
    # image.save(SAVE_FILE, format="PNG")
    # np_img = np.uint8(np_img)
    sam2_predictor.set_image(image)
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

    # only do this if there are detections
    if (input_boxes.shape != (0,4)):

        # unessecary call
        # sam2_predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_boxes,
        #     multimask_output=False,
        # )


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




@app.post("/get_batched_embeddings")
async def get_arr_embeddings(img_json: ArrayInput):

    # print("Calling get_batched_embeddings")

    # convert each image into a numpy array, np.float32
    imgs = [np.array(img).astype(np.float32) for img in img_json.img_arr]
    # with open('/home/davin123/LOTUS-PARTS/Lotus-Parts/lotus/skill_learning/Grounded-SAM-2/api_buffer.json', 'w') as file:
    #    imgs = json.load(file)

<<<<<<< HEAD
    # print("GET embeddings")
    batched_img_embeddings, batched_orig_hw = sam2_predictor.image_embeddings_batch(imgs)
=======
    print("GET embeddings")
    batched_img_embeddings = sam2_predictor.image_embeddings_batch(imgs)
>>>>>>> parent of 0dc8871 (endpoint works for detections now)

    # print("Got imbeddings")

    # input boxes for dense and sparse embeddings
    input_boxes_for_batching = []
    # for img in imgs:
    for img in img_json.img_arr:
        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot
        text = img_json.prompt

        # prepare the image 
        np_img = np.array(img)
        image = Image.fromarray(np.uint8(np_img)).convert('RGB')
        # image.save(SAVE_FILE, format="PNG")
        # np_img = np.uint8(np_img)
        sam2_predictor.set_image(image)
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

        input_boxes_for_batching.append(input_boxes)

    # print("Got boxes")

    # pass in the image_embeddings and also pass in the number of images
    f_embeddings = sam2_predictor.get_sparse_and_dense_embeddings_batch(
        point_coords_batch=None,
        point_labels_batch=None,
        box_batch=input_boxes_for_batching,
        mask_input_batch=None,
        image_embeddings=batched_img_embeddings,
        num_of_images=len(imgs),
        multimask_output=False,
    )

    # convert into a list 
<<<<<<< HEAD
    # print("converting to list")

    # for array in f_embeddings:
    #    print("SHAPE RETURN: ", array.shape)

    f_embeddings_converted = [f_embed.tolist() for f_embed in f_embeddings]
    # print("converted")

    # for list in f_embeddings_converted:
    #    print("List length: ", len(list))
=======
    print("converting to list")
    f_embeddings_converted = [f_embed.tolist() for f_embed in f_embeddings]
    print("converted")
>>>>>>> parent of 0dc8871 (endpoint works for detections now)
    
    return {"embeddings": f_embeddings_converted}



# Here we will be working with gPRC to use HTTP/2 
# this should speed up the runtime by about 5-7 times 
# esepcially in cases where you have a lot of data 
# and in this case this is especially true given that we 
# have a lot of image data 
