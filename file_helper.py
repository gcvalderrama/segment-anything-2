import logging
import os
import shutil
import cv2
import numpy as np

def delete_folders(worker_dir, name):
    shutil.rmtree(f'{worker_dir}/{name}', ignore_errors=True)        


def create_folders(worker_dir, name):
    os.makedirs(f'{worker_dir}/{name}', exist_ok=True)

def save_mask(image_np, mask):
    mask_uint8 = np.zeros((100, 100), dtype=np.uint8)         
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)    
    mask_uint8 = (mask * 255).astype(np.uint8) 
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cut_image = cut_image_using_contours(image_cv2, contours)
    return cut_image  

def save_image(file_output, cut_image):
    cv2.imwrite(file_output, cut_image)

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
    color_background = np.full_like(image, white_color)

    cut_image_with_white_bg = np.where(mask_3channel == 0, color_background, cut_image)

    # Display the results
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Cut Image', cut_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cut_image_with_white_bg
