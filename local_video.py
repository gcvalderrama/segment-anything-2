import os
import logging
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from file_helper import create_folders, delete_folders, save_mask, save_image

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format in log messages
)

# select the device for computation
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


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def print_key_frame(ann_frame_idx, video_input_dir, points):
    labels = np.ones(len(points), dtype=int)     
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_input_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())    
    plt.show()

def set_key_frame(ann_frame_idx, ann_obj_id, points, logs=False, video_input_dir=None ):
    labels = np.ones(len(points), dtype=int)     
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    if logs:
        labels = np.ones(len(points), dtype=int)     
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_input_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()

    return out_obj_ids, out_mask_logits

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

if __name__ == "__main__":
    
    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`

    video_dir = "E:/aliceplace/"
    target_name = "C0345"
    video_input_dir = f"{video_dir}/{target_name}"
    delete_folders(video_dir, f"{target_name}_track")
    # scan all the JPEG frame names in this directory    
    frame_names = [
        p
        for p in os.listdir(video_input_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))    
    
    point_x = 1400
    point_y = 200
    check_point_a = [[point_x, point_y], [point_x, point_y + 500], [point_x, point_y + 1000]]   
    check_point_b = [[point_x, point_y], [point_x, point_y + 500], [point_x, point_y + 1000]]
    points = np.array(check_point_a, dtype=np.float32)        
    print_key_frame(ann_frame_idx=0, video_input_dir=video_input_dir, points=points)    
    points = np.array(check_point_b, dtype=np.float32)        
    print_key_frame(ann_frame_idx=2, video_input_dir=video_input_dir, points=points)
    
    
    inference_state = predictor.init_state(video_path=video_input_dir)
    set_key_frame(ann_frame_idx=0, ann_obj_id=0, points=check_point_a)
    set_key_frame(ann_frame_idx=2, ann_obj_id=224, points=check_point_b)    
    create_folders(video_dir, f"{target_name}_track")

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")

    logging.info("video segments len %s", len(video_segments))    
    logging.info("video segments keys %s", list(video_segments.keys())[:10])    

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #plt.figure(figsize=(6, 4))
        #plt.title(f"frame {out_frame_idx}")
        #plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

        temp_image = np.array(Image.open(os.path.join(video_input_dir, frame_names[out_frame_idx])).convert("RGB"))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            print(out_obj_id)
            mask = np.squeeze(out_mask)            
            cut_image = save_mask(temp_image, mask)
            save_image(f"{video_dir}/{target_name}_track/{frame_names[out_frame_idx]}", cut_image)
            #show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            #plt.show()