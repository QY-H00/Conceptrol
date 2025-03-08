import torch
from typing import Union, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
import numpy as np
import cv2

condition_dict = {
    "depth": 0,
    "canny": 1,
    "subject": 4,
    "coloring": 6,
    "deblurring": 7,
    "fill": 9,
}


class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        mask=None,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"

    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Returns the condition image.
        """
        if condition_type == "depth":
            from transformers import pipeline

            depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cuda",
            )
            source_image = raw_img.convert("RGB")
            condition_img = depth_pipe(source_image)["depth"].convert("RGB")
            return condition_img
        elif condition_type == "canny":
            img = np.array(raw_img)
            edges = cv2.Canny(img, 100, 200)
            edges = Image.fromarray(edges).convert("RGB")
            return edges
        elif condition_type == "subject":
            return raw_img
        elif condition_type == "coloring":
            return raw_img.convert("L").convert("RGB")
        elif condition_type == "deblurring":
            condition_image = (
                raw_img.convert("RGB")
                .filter(ImageFilter.GaussianBlur(10))
                .convert("RGB")
            )
            return condition_image
        elif condition_type == "fill":
            return raw_img.convert("RGB")
        return self.condition

    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[condition_type]

    def _encode_image(self, pipe: FluxPipeline, cond_img: Image.Image) -> torch.Tensor:
        """
        Encodes an image condition into tokens using the pipeline.
        """
        cond_img = pipe.image_processor.preprocess(cond_img)
        cond_img = cond_img.to(pipe.device).to(pipe.dtype)
        cond_img = pipe.vae.encode(cond_img).latent_dist.sample()
        cond_img = (
            cond_img - pipe.vae.config.shift_factor
        ) * pipe.vae.config.scaling_factor
        cond_tokens = pipe._pack_latents(cond_img, *cond_img.shape)
        cond_ids = pipe._prepare_latent_image_ids(
            cond_img.shape[0],
            cond_img.shape[2],
            cond_img.shape[3],
            pipe.device,
            pipe.dtype,
        )
        return cond_tokens, cond_ids

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_type in [
            "depth",
            "canny",
            "subject",
            "coloring",
            "deblurring",
            "fill",
        ]:
            tokens, ids = self._encode_image(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, type_id
