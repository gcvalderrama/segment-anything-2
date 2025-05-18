import uuid
import zmq
import cv2
import logging
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor        
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    #filename='detectron_agent.log',
    #filemode='a'
)

import sys
import os
import json
import warnings
# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", message=".*CuDNN attention kernel not used because.*")
warnings.filterwarnings("ignore", message=".*Memory efficient.*")
warnings.filterwarnings("ignore", message=".*Flash attention kernel not used because.*")
warnings.filterwarnings("ignore", message=".*Expected query, key and value.*")
warnings.filterwarnings("ignore", message=".*Flash Attention kernel failed.*")




sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    

def ensure_model():    
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def read_information(directory_path, key_name):    
    file_name = os.path.join(directory_path, key_name + ".detect.txt")
    with open(file_name, "r") as f:
        information = json.loads(f.read())
    return information

def extract_humans(image, information: dict, labels: list = ['person']):        
    image_width, image_height = image.size
    image_area = image_width * image_height
    boxes = []
    for key, items in information.items():                
        for item in items:                                                            
            box = item['box']
            score = item['score']
            area = (box[2] - box[0]) * (box[3] - box[1])                
            if key not in labels or score < 0.90 or area < image_area * 0.005:   
                logging.debug(f"Skip {key} {score} area of the box:  {box} :  {area}")                             
                continue
            else:                
                boxes.append(box)    
    return boxes

def cut_image_using_contours_v2(image, contours):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    mask_binary = mask == 0
    mask_binary_3ch = np.stack([mask_binary]*3, axis=-1)

    color_background = np.full_like(image, (50, 205, 50))  # BGR for "green lemon" black_color = (0, 0, 0)  white_color = (255, 255, 255)
    cut_image = cv2.bitwise_and(image, image, mask=mask)
    cut_image_with_bg = np.where(mask_binary_3ch, color_background, cut_image)

    return cut_image_with_bg

def save_mask(image_np, mask):
    mask_uint8 = np.zeros((100, 100), dtype=np.uint8)         
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)    
    mask_uint8 = (mask * 255).astype(np.uint8) 
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cut_image = cut_image_using_contours_v2(image_cv2, contours)
    output_rgb = cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB)
    return output_rgb



def segment_picture(predictor, image_opened, boxes: list):    
    logging.info(f"segment using device: {device}")    
    logging.info(f"boxes: {boxes}")    
    image = np.array(image_opened.convert("RGB"))
    predictor.set_image(image)         
    all_masks, all_scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=True,
    )    
    fused_masks = []
        
    if all_masks.ndim == 3:        
        all_masks = np.expand_dims(all_masks, axis=0)
    if all_scores.ndim == 1:
        all_scores = np.expand_dims(all_scores, axis=0)
        
    logging.info(all_masks.shape)
    logging.info(all_scores)
    
    for masks, scores in zip(all_masks, all_scores):                
        sorted_ind = np.argsort(scores)[::-1]        
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]        
        fused_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for i in range(min(3, len(masks))):
            fused_mask = np.maximum(fused_mask, masks[i])
        fused_masks.append(fused_mask)
        
    combined_mask = np.zeros_like(fused_masks[0], dtype=np.uint8)
    for mask in fused_masks:
        combined_mask = np.maximum(combined_mask, mask)
    
    return combined_mask
            
def save_segmented_image(image_opened, key_term, mask, output_directory):
    image = np.array(image_opened.convert("RGB"))    
    mask_path = os.path.join(output_directory, f"{key_term}.segment.npz")
    np.savez_compressed(mask_path, mask)
    logging.info(f"Fused mask saved in compressed format to {mask_path}")    
    
    cut_image = save_mask(image, mask)                      
    output_path =os.path.join(output_directory, f"{key_term}.jpeg" )
    cv2.imwrite(output_path, cv2.cvtColor(cut_image, cv2.COLOR_RGB2BGR))
    logging.info(f"Segmented image saved to {output_path}")    
    
    

def load_mask(image_path, mask_path):
    combined_mask = np.load(mask_path)['arr_0']
    print(combined_mask)    
    image = Image.open(image_path)     
    image = np.array(image.convert("RGB"))    
    
    cut_image = save_mask(image, combined_mask)                  
    output_path =f"./wip/salida.jpeg"
    cv2.imwrite(output_path, cv2.cvtColor(cut_image, cv2.COLOR_RGB2BGR))
    logging.info(f"Segmented image saved to {output_path}")    
        
# python start_agent.py '/mnt/eight/shared/alice/C0761/frames' '/mnt/eight/shared/alice/C0761/segment' 3 0
if __name__ == "__main__":                
    folder_path = "./wip"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)
            
    directory = sys.argv[1] if len(sys.argv) > 1 else '/mnt/eight/shared/alice/C0761/frames'
    output_directory = sys.argv[2] if len(sys.argv) > 2 else '/mnt/eight/shared/alice/C0761/segments'
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    rest = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    logging.info(f"directory: {directory}")
    logging.info(f"seed: {seed}, rest: {rest}")
    predictor = ensure_model()
    # Iterate through all files in the directory and process images
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_name_no_ext, extension = os.path.splitext(file_name)
            if int(file_name_no_ext) % seed == rest:                
                image = None
                try:                    
                    image = Image.open(os.path.join(directory, file_name))                                        
                    information = read_information(directory, file_name_no_ext)                
                    boxes = extract_humans(image, information)
                    if boxes:                    
                        mask = segment_picture(predictor, image, boxes) 
                        save_segmented_image(image, file_name_no_ext, mask, output_directory)                        
                    
                finally:
                    if image:
                        image.close()                                