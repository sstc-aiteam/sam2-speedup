"""Sanity-check that fp16+compile predictions match the fp32 eager baseline."""
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CONFIG = "configs/sam2/sam2_hiera_s.yaml"
CKPT = "checkpoints/sam2_hiera_small.pt"
IMAGE_PATH = "notebooks/images/truck.jpg"
POINT = np.array([[500, 375]])
LABEL = np.array([1])


def run(dtype, compile_encoder):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sam2_model = build_sam2(CONFIG, CKPT, device="cuda")
    if compile_encoder:
        sam2_model.image_encoder.forward = torch.compile(sam2_model.image_encoder.forward)
    predictor = SAM2ImagePredictor(sam2_model)
    image = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    ctx = torch.autocast("cuda", dtype=dtype) if dtype else torch.autocast("cuda", enabled=False)
    with torch.inference_mode(), ctx:
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=POINT, point_labels=LABEL, multimask_output=True
        )
    return masks, scores


if __name__ == "__main__":
    masks_fp32, scores_fp32 = run(None, False)
    masks_fp16c, scores_fp16c = run(torch.float16, True)

    print("fp32 scores:  ", scores_fp32)
    print("fp16+compile: ", scores_fp16c)

    best_fp32 = masks_fp32[scores_fp32.argmax()].astype(bool)
    best_fp16c = masks_fp16c[scores_fp16c.argmax()].astype(bool)

    intersection = np.logical_and(best_fp32, best_fp16c).sum()
    union = np.logical_or(best_fp32, best_fp16c).sum()
    iou = intersection / union
    print(f"best-mask IoU (fp32 vs fp16+compile): {iou:.4f}")
    print(f"score diff: {abs(scores_fp32.max() - scores_fp16c.max()):.4f}")
