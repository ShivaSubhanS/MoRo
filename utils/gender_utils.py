"""
Gender prediction using YOLOv8n-cls model from:
  DhanushSGowda/yolov8n-gender-classification
Runs on CPU to keep GPU free.
"""
import os
import cv2
import numpy as np
from collections import Counter


MODEL_REPO = "DhanushSGowda/yolov8n-gender-classification"
MODEL_PATH = "ckpt/gender_cls.pt"


def _download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[Gender] Downloading gender classifier to {MODEL_PATH} ...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename="best_Gender_classification.pt")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    import shutil
    shutil.copy(path, MODEL_PATH)
    print(f"[Gender] Model saved to {MODEL_PATH}")


def _crop_person(img, cx, cy, scale, margin=1.2):
    """Crop person from image given bbx_xys format (cx, cy, scale)."""
    half = scale * 100.0 * margin
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(img.shape[1], int(cx + half))
    y2 = min(img.shape[0], int(cy + half))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def predict_gender(image_paths, bbx_xys, track_idx=0, num_samples=10):
    """
    Predict gender for a given track by sampling frames and voting.

    Args:
        image_paths: list of image paths (L,)
        bbx_xys: tensor (N, L, 3) - cx, cy, scale per track per frame
        track_idx: which track to classify
        num_samples: number of frames to sample for voting

    Returns:
        str: 'male' or 'female'
    """
    from ultralytics import YOLO

    _download_model()
    model = YOLO(MODEL_PATH)
    model.to("cpu")

    L = len(image_paths)
    # Sample evenly across the video
    sample_indices = np.linspace(0, L - 1, min(num_samples, L), dtype=int).tolist()

    votes = []
    for idx in sample_indices:
        img_bgr = cv2.imread(image_paths[idx])
        if img_bgr is None:
            continue
        cx, cy, scale = bbx_xys[track_idx, idx].cpu().numpy()
        crop = _crop_person(img_bgr, cx, cy, scale)
        if crop is None or crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        results = model.predict(source=crop_rgb, imgsz=224, device="cpu", verbose=False)
        for result in results:
            top1_idx = result.probs.top1
            label = result.names[top1_idx].lower()
            # check female first ("female" contains "male" as substring)
            if "female" in label:
                votes.append("female")
            elif "male" in label:
                votes.append("male")

    del model

    if not votes:
        print("[Gender] Could not determine gender, defaulting to male")
        return "male"

    gender = Counter(votes).most_common(1)[0][0]
    confidence = votes.count(gender) / len(votes)
    print(f"[Gender] Predicted: {gender} (confidence: {confidence:.0%}, votes: {len(votes)})")
    return gender
