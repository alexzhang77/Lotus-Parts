import grpc
from concurrent import futures
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import sam2_service_pb2
import sam2_service_pb2_grpc

# Initialize models and predictors as in your original code
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "./request_image.png"

torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
sam2_predictor._is_batch = True

processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

class SAM2ServiceServicer(sam2_service_pb2_grpc.SAM2ServiceServicer):
    def GetBatchedEmbeddings(self, request, context):
        # Convert request images to numpy arrays
        imgs = [np.array(request.img_arr).astype(np.float32)]
        text = request.prompt

        # Get image embeddings
        batched_img_embeddings, batched_orig_hw = sam2_predictor.image_embeddings_batch(imgs)
        input_boxes_for_batching = []

        for img in imgs:
            np_img = np.array(img)
            image = Image.fromarray(np.uint8(np_img)).convert("RGB")
            # image.save(SAVE_FILE, format="PNG")
            sam2_predictor.set_image(image)

            # Process image with Grounding DINO
            inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
            )
            input_boxes = results[0]["boxes"].cpu().numpy()
            input_boxes_for_batching.append(input_boxes)

        # Get sparse and dense embeddings
        f_embeddings = sam2_predictor.get_sparse_and_dense_embeddings_batch(
            point_coords_batch=None,
            point_labels_batch=None,
            box_batch=input_boxes_for_batching,
            mask_input_batch=None,
            image_embeddings=batched_img_embeddings,
            num_of_images=len(imgs),
            orig_hw=batched_orig_hw,
            multimask_output=False,
        )

        # Convert embeddings to response
        embeddings = [sam2_service_pb2.Embedding(values=embedding.tolist()) for embedding in f_embeddings]
        return sam2_service_pb2.ArrayOutput(embeddings=embeddings)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sam2_service_pb2_grpc.add_SAM2ServiceServicer_to_server(SAM2ServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
