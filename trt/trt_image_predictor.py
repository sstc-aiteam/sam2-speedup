"""SAM2ImagePredictor subclass that runs the image encoder and mask decoder
through TensorRT engines instead of the PyTorch model.

Only overrides set_image() and _predict() -- the two calls that touch the
neural net. Everything else (SAM2Transforms preprocessing/postprocessing,
reset_predictor(), and critically all of SAM2AutomaticMaskGenerator's
crop/batch/NMS/stability-score/RLE orchestration in
sam2/automatic_mask_generator.py, which only calls into a predictor through
these same methods) is reused unchanged. See trt/export_encoder.py and
trt/export_decoder.py for exactly what each engine computes and the scope
this decoder engine is limited to (point-only prompts, multimask_output=True,
single image) -- this predictor asserts on both to stay honest with that
scope rather than silently producing wrong output for unsupported calls.
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trt_runner import TRTModule  # noqa: E402

from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


class TRTSAM2ImagePredictor(SAM2ImagePredictor):
    def __init__(
        self,
        sam_model,
        encoder_engine_path: str,
        decoder_engine_path: str,
        **kwargs,
    ):
        super().__init__(sam_model, **kwargs)
        self._trt_encoder = TRTModule(encoder_engine_path, device=str(self.device))
        self._trt_decoder = TRTModule(decoder_engine_path, device=str(self.device))

    @torch.no_grad()
    def set_image(self, image) -> None:
        self.reset_predictor()
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        else:
            from PIL.Image import Image as PILImage

            if isinstance(image, PILImage):
                w, h = image.size
                self._orig_hw = [(h, w)]
            else:
                raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)[None, ...].to(self.device)
        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"

        out = self._trt_encoder(image=input_image)
        self._features = {
            "image_embed": out["image_embed"],
            "high_res_feats": [out["high_res_feat0"], out["high_res_feat1"]],
        }
        self._is_image_set = True

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        assert boxes is None and mask_input is None, (
            "the TRT decoder engine was exported for point-only prompts with "
            "no mask_input refinement (the SAM2AutomaticMaskGenerator path, "
            "use_m2m=False) -- see trt/export_decoder.py"
        )
        assert multimask_output, (
            "the TRT decoder engine was exported with multimask_output=True "
            "baked in -- see trt/export_decoder.py"
        )
        assert point_coords is not None

        out = self._trt_decoder(
            image_embed=self._features["image_embed"],
            high_res_feat0=self._features["high_res_feats"][0],
            high_res_feat1=self._features["high_res_feats"][1],
            point_coords=point_coords.to(torch.float32),
            point_labels=point_labels.to(torch.int32),
        )
        low_res_masks = out["low_res_masks"]
        iou_predictions = out["iou_predictions"]

        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks
