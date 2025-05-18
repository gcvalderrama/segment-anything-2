from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor        
from PIL import Image
import numpy as np
import os
import json
import torch
import cv2
import logging
import argparse
import sys
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format in log messages
    stream=sys.stdout  # Write log messages to standard output
)

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        default="human",
        help="input human pictures",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        default="segment",
        help="output human pictures",
    )
    return parser.parse_args()


def ensure_model():    
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor
    
def generate_points(box: list):
    ax, ay , bx, by = box
    x_delta = int(bx - ax)
    y_delta = int(by - ay)
    zx = ax + int(x_delta /2 )
    zy = ay + int(y_delta /2 )
    # zy2 = ay + int(y_delta/3 ) 
    # zy3 = ay + int(y_delta/3 ) * 2
    # zy4 = ay + int(y_delta/5 ) * 4
    # zy5 = ay + int(y_delta/5 ) * 1        
    points = [[zx, zy]]
    return points    
    
def segment_picture(predictor, image_path, points):        
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)                

    input_point = np.array(points)    
    input_label = np.array([1] * len(points), dtype=int)
    LOGGER.debug(f"input point: {input_point}")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # self.show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()  
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # Fusion 3 masks into one mask
    fused_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for i in range(min(3, len(masks))):
        fused_mask = np.maximum(fused_mask, masks[i])
            

    cut_image = save_mask(image, fused_mask)              
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # self.show_mask(fused_mask, plt.gca(), borders=True)
    # plt.title("Fused Mask", fontsize=18)
    # plt.axis('off')
    # plt.show()
    return cut_image        
    
def segment_directory(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory")        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    LOGGER.info(f"segmenting directory {input_dir}")
    predictor = ensure_model()
    catalog = json.load(open(os.path.join(input_dir, "catalog.json"), "r"))        
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".jpeg"):                                
            file_key, ext = os.path.splitext(filename)
            try:                    
                if file_key in catalog:
                    box = catalog[file_key]
                    LOGGER.info(f"segmenting {filename} {box}")
                    points = generate_points(catalog[file_key])
                    LOGGER.info(f"segmenting {filename} {points}")
                    result = segment_picture( predictor, file_path, points)
                    output_file = os.path.join(output_dir, filename)
                    cv2.imwrite(output_file, result)
                    LOGGER.info(f"saved {output_file}")
                else:
                    LOGGER.info(f"no key found {file_key}")
            except:
                LOGGER.info(f"error processing {filename}")
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            # plt.title("Cut Image with White Background", fontsize=18)
            # plt.axis('off')
            # plt.show()
            
            
        else:
            LOGGER.info(f"no file found {filename}")

def save_mask(image_np, mask):
    mask_uint8 = np.zeros((100, 100), dtype=np.uint8)         
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)    
    mask_uint8 = (mask * 255).astype(np.uint8) 
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cut_image = cut_image_using_contours(image_cv2, contours)
    return cut_image 

def cut_image_using_contours(image, contours):
    # Convert image to grayscale if it is not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Convert the mask to 3 channels to match the image
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the original image
    cut_image = cv2.bitwise_and(image, mask_3channel)
    
    green_lemon_color = (50, 205, 50)
    black_color = (0, 0, 0)
    white_color = (255, 255, 255)
    color_background = np.full_like(image, green_lemon_color)

    cut_image_with_white_bg = np.where(mask_3channel == 0, color_background, cut_image)

    # Display the results
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Cut Image', cut_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cut_image_with_white_bg



def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:    
    return args

if __name__ == "__main__":
    args = validate_inputs(parse_args())        
    logging.info(f"segment using device: {device}")
    logging.info(f"input_dir: {args.input_dir}")    
    logging.info(f"output_dir: {args.output_dir}")
    segment_directory(args.input_dir, args.output_dir)
    exit(0)
    
    