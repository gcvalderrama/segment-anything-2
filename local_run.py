import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2


np.random.seed(3)

# region print
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
        green_lemon_background = np.full_like(image, green_lemon_color)

        white_color = (0, 0, 0)
        white_background = np.full_like(image, white_color)

        cut_image_with_white_bg = np.where(mask_3channel == 0, white_background, cut_image)

        # Display the results
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Mask', mask)
        # cv2.imshow('Cut Image', cut_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return cut_image_with_white_bg

        # Optionally, save the cut image
        # cv2.imwrite('cut_image.jpg', cut_image)

def save_mask(image_np, mask):
    mask_uint8 = np.zeros((100, 100), dtype=np.uint8)         
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cut_image = cut_image_using_contours(image_cv2, contours)
    return cut_image    

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:        
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)





def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

# endregion


def use_box(image, device):
    input_box = np.array([0, 0, 1200, 1200])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)
        print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        show_masks(image, masks, scores, box_coords=input_box)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
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

    file_path = 'E:/aliceplace/C0083/0032.jpeg'
    file_output = 'E:/aliceplace/C0083_body/0032.jpeg'

    raw_image = Image.open(file_path).convert("RGB")

    image = Image.open(file_path)
    image = np.array(image.convert("RGB"))
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()   

    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    

    input_point = np.array([[650, 550]])
    input_label = np.array([1])

    #use_box(image, device)

    

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()  
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)
        print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        print(masks.shape)
        borders=True
        point_coords=input_point
        input_labels=input_label

    

        for i, (mask, score) in enumerate(zip(masks, scores)):
            print("mask")
            print(mask.shape) # (2160, 3840) (1, 2160, 3840)
            print(type(mask))
            #cut_image = save_mask(image, mask)
            # cv2.imwrite(file_output, cut_image)
            #plt.figure(figsize=(10, 10))
            #plt.imshow(image)
            #show_mask(mask, plt.gca(), borders=borders)
            # if point_coords is not None:
            #     assert input_labels is not None
                #show_points(point_coords, input_labels, plt.gca())            
            # if len(scores) > 1:
            #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            #plt.axis('off')
            #plt.show()
            break

    #show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)



#     predictor.set_image(<your_image>)
#     masks, _, _ = predictor.predict(<input_prompts>)