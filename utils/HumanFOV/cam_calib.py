import torch
import cv2
import numpy as np
from .fl_net import FLNet

from tqdm import tqdm

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

def estimate_focal_length(image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FLNet()
    checkpoint = torch.load("ckpt/cam_model_cleaned.ckpt", weights_only=False)["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    fl_list = []
    for img_path in tqdm(image_paths, desc="Estimating focal lengths"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1)) / 255.0
        img_full_resized = torch.from_numpy(img_full_resized).float()
        # torchvision.transforms.Normalize(mean, std) equivalent (no torchvision import)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_full_resized.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img_full_resized.dtype).view(3, 1, 1)
        img_full_resized = (img_full_resized - mean) / std
        img_full_resized = img_full_resized.to(device)

        estimated_fov, _ = model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        fl_list.append(fl_h)
    return fl_list
