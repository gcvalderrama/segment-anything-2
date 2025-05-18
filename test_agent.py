import json
import random
import sys
import matplotlib.pyplot as plt
import os
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
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    #filename='detectron_agent.log',
    #filemode='a'
)

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

def save_mask(image_np, mask):
    mask_uint8 = np.zeros((100, 100), dtype=np.uint8)         
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)    
    mask_uint8 = (mask * 255).astype(np.uint8) 
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cut_image = cut_image_using_contours(image_cv2, contours)
    return cut_image 


    
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
    points = []
    num_points = 100
    radius_increment = 5
    angle_increment = 0.1

    for i in range(num_points):
        radius = i * radius_increment
        angle = i * angle_increment
        x = zx + int(radius * np.cos(angle))
        y = zy + int(radius * np.sin(angle))
        if ax <= x <= bx and ay <= y <= by:  # Ensure points are within the box
            points.append((x, y))
 
    
    return points 

def drawn_boxes(img, label, box):    
    draw = ImageDraw.Draw(img)   
        
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
    random_color = random.choice(colors)
    draw.rectangle(box, outline=random_color, width=3)                        
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
    text_size = draw.textlength(label, font=font)
    text_position = (box[0], box[1] - text_size)        
    draw.rectangle(
        [text_position, (text_position[0] + text_size, text_position[1] + text_size)],
        fill="red"
    )
    draw.text(text_position, label, fill="white", font=font)           

def draw_points_v2(img, point):
    draw = ImageDraw.Draw(img)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
    random_color = random.choice(colors)
    for p in point:
        draw.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill=random_color)
    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    
def get_id_by_box(image_path, box):
    file_name = os.path.splitext(os.path.basename(image_path))[0]    
    return f"{file_name}." + ''.join(str(int(b)) for b in box)

def segment_picture_by_boxes(predictor, image_path, boxes: list):
    logging.info(f"segment using device: {device}")
    logging.info(f"input image: {image_path}")
    logging.info(f"input boxes: {boxes}")
    image = Image.open(image_path)     
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)      
    
    if boxes ==  [[3015.89, 1841.56, 3309.18, 2640.42], [4365.72, 1884.06, 4621.9, 2531.71]]:
        print("test")
        return      
    else:
        print(type(boxes))
        print(boxes)
    
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(boxes),
        multimask_output=True,
    )
    print(masks.shape)
    
    # Combine all masks into one mask
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)
    
    print(combined_mask.shape)
    
    # (10, 1, 4000, 6000)
    
    # sorted_ind = np.argsort(scores)[::-1]
    # masks = masks[sorted_ind]
    # scores = scores[sorted_ind]
    # logits = logits[sorted_ind]
    # # Fusion 3 masks into one mask
    fused_mask = np.zeros_like(combined_mask[0], dtype=np.uint8)
    for i in range(min(3, len(combined_mask))):
        fused_mask = np.maximum(fused_mask, combined_mask[i])
        
    object_id = get_id_by_box(image_path, box)
    # Save the fused mask as a compressed .npz file
    compressed_mask_path = f"./wip/{object_id}.npz"    
    np.savez_compressed(compressed_mask_path, fused_mask)
    logging.info(f"Fused mask saved in compressed format to {compressed_mask_path}")

    cut_image = save_mask(image, fused_mask)                  
    output_path =f"./wip/{object_id}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(cut_image, cv2.COLOR_RGB2BGR))
    logging.info(f"Segmented image saved to {output_path}")
    
    return cut_image
    
def segment_picture(predictor, image_path, box: list):
    points = generate_points(box)    
    logging.info(f"segment using device: {device}")
    logging.info(f"input image: {image_path}")
    logging.info(f"input points: {points}")
    
    image = Image.open(image_path)     
    
    input_point = np.array(points)    
    input_label = np.array([1] * len(points), dtype=int)
    
    #drawn_boxes(image, "person", box)
    #draw_points_v2(image, points)
    #image.save("wip/test.jpg")    
    
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)            
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True,
    # )
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
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
        
    object_id = get_id_by_box(image_path, box)
    # Save the fused mask as a compressed .npz file
    compressed_mask_path = f"./wip/{object_id}.npz"    
    np.savez_compressed(compressed_mask_path, fused_mask=fused_mask)
    logging.info(f"Fused mask saved in compressed format to {compressed_mask_path}")

    cut_image = save_mask(image, fused_mask)                  
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # self.show_mask(fused_mask, plt.gca(), borders=True)
    # plt.title("Fused Mask", fontsize=18)
    # plt.axis('off')
    # plt.show()
    
    # Save the resulting cut image as an image file
    output_path =f"./wip/{object_id}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(cut_image, cv2.COLOR_RGB2BGR))
    logging.info(f"Segmented image saved to {output_path}")
    
    return cut_image        

def read_information(image_path):
    directory_path = os.path.dirname(image_path)
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    file_name = os.path.join(directory_path, file_name + ".detect.txt")
    with open(file_name, "r") as f:
        information = json.loads(f.read())
    return information
    
if __name__ == "__main__":            
    predictor = ensure_model()
    image_path = '/mnt/eight/shared/alice/BOTE/frames/DSC05954.JPG'
    information = read_information(image_path)
    
    folder_path = "./wip"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)
    
    print(information)
    image = Image.open(image_path)
    image_width, image_height = image.size
    image_area = image_width * image_height
    logging.info(f"Image dimensions: {image_width}x{image_height}, Area: {image_area}")
    boxes = []
    for key, items in information.items():                
        for item in items:                                                            
            box = item['box']
            score = item['score']
            area = (box[2] - box[0]) * (box[3] - box[1])                
            if score < 0.90 or area < image_area * 0.005:   
                logging.debug(f"Skip {key} {score} area of the box:  {box} :  {area}")                             
                continue
            else:
                logging.info(f"Calculated {key} {score} area of the box:  {box} :  {area}")                
                boxes.append(box)
    print(boxes)
    result = segment_picture_by_boxes(predictor, image_path, boxes)                   