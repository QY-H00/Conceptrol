import os
from typing import List

import torch
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTokenizer,
)

from .attention_processor import (
    AttnProcessor,
    CNAttnProcessor,
    IPAttnProcessor,
    ConceptrolAttnProcessor,
)
from .resampler import Resampler
from .utils import get_generator
from huggingface_hub import hf_hub_download

SD_CONCEPT_LAYER = ["up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor"]
SDXL_CONCEPT_LAYER = ["up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor"]


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(
        self,
        cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    ):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():  # noqa: SIM118
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(
                        CNAttnProcessor(num_tokens=self.num_tokens)
                    )
            else:
                self.pipe.controlnet.set_attn_processor(
                    CNAttnProcessor(num_tokens=self.num_tokens)
                )

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = (
                            f.get_tensor(key)
                        )
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                            f.get_tensor(key)
                        )
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=torch.float16)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(clip_image_embeds)
        )
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_images=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if pil_images is not None else clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image in pil_images:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
                1, num_samples, 1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat(
                [prompt_embeds_, *image_prompt_embeds_list], dim=1
            )
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds_, *uncond_image_prompt_embeds_list], dim=1
            )

        # generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            # generator=generator,
            **kwargs,
        ).images

        return images


class ConceptrolIPAdapter:
    def __init__(
        self,
        sd_pipe,
        image_encoder_path,
        ip_ckpt,
        device,
        num_tokens=4,
        global_masking=True,
        adaptive_scale_mask=False,
    ):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter(global_masking, adaptive_scale_mask)

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(self.device, dtype=torch.float16)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self, global_masking, adaptive_scale_mask):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():  # noqa: SIM118
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = ConceptrolAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    name=name,
                    global_masking=global_masking,
                    adaptive_scale_mask=adaptive_scale_mask,
                    concept_mask_layer=SD_CONCEPT_LAYER,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        for name in unet.attn_processors.keys():  # noqa: SIM118
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if cross_attention_dim is not None:
                unet.attn_processors[name].set_global_view(unet.attn_processors)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(
                        CNAttnProcessor(num_tokens=self.num_tokens)
                    )
            else:
                self.pipe.controlnet.set_attn_processor(
                    CNAttnProcessor(num_tokens=self.num_tokens)
                )

    def load_ip_adapter(self):
        ckpt_path = self.ip_ckpt
        # If the checkpoint path is not an existing file and is not a full URL,
        # assume it's a Huggingface repository specification.
        if not os.path.exists(self.ip_ckpt) and not self.ip_ckpt.startswith("http"):
            # If a colon is present, use it to split repo_id and filename.
            if ":" in self.ip_ckpt:
                repo_id, filename = self.ip_ckpt.split(":", 1)
            else:
                parts = self.ip_ckpt.split("/")
                if len(parts) > 2:
                    # For example, "h94/IP-Adapter/models/ip-adapter-plus_sd15.bin"
                    # repo_id becomes "h94/IP-Adapter" and filename "models/ip-adapter-plus_sd15.bin".
                    repo_id = "/".join(parts[:2])
                    filename = "/".join(parts[2:])
                else:
                    repo_id = self.ip_ckpt
                    filename = "models/ip-adapter-plus_sd15.bin"  # default filename if not specified
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Load the state dictionary from the checkpoint file.
        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = (
                            f.get_tensor(key)
                        )
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                            f.get_tensor(key)
                        )
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load the state dictionaries into the corresponding models.
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(
                images=pil_image, return_tensors="pt"
            ).pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=torch.float16)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(
            torch.zeros_like(clip_image_embeds)
        )
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, ConceptrolAttnProcessor):
                attn_processor.scale = scale

    def load_textual_concept(self, prompt, subjects):
        tokens = self.tokenizer.tokenize(prompt)
        textual_concept_idxs = []
        offset = 1  # TODO: change back to 1 if not true

        for subject in subjects:
            subject_tokens = self.tokenizer.tokenize(subject)
            start_idx = tokens.index(subject_tokens[0]) + offset
            end_idx = tokens.index(subject_tokens[-1]) + offset
            textual_concept_idxs.append((start_idx, end_idx + 1))
            print("Locate:", subject, start_idx, end_idx + 1)

        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, ConceptrolAttnProcessor):
                attn_processor.textual_concept_idxs = textual_concept_idxs

    def generate(
        self,
        pil_images=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=42,
        subjects=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1  # not support multiple prompts

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if subjects:
            self.load_textual_concept(prompt, subjects)
        else:
            raise ValueError("Subjects must be provided")

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image in pil_images:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
                1, num_samples, 1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat(
                [prompt_embeds_, *image_prompt_embeds_list], dim=1
            )
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds_, *uncond_image_prompt_embeds_list], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_images,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1  # not support multiple prompts

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image in pil_images:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
                1, num_samples, 1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, *image_prompt_embeds_list], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, *uncond_image_prompt_embeds_list], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class ConceptrolIPAdapterXL(ConceptrolIPAdapter):
    """SDXL"""

    def set_ip_adapter(self, global_masking, adaptive_scale_mask):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():  # noqa: SIM118
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = ConceptrolAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                    name=name,
                    global_masking=global_masking,
                    adaptive_scale_mask=adaptive_scale_mask,
                    concept_mask_layer=SDXL_CONCEPT_LAYER,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        for name in unet.attn_processors.keys():  # noqa: SIM118
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if cross_attention_dim is not None:
                unet.attn_processors[name].set_global_view(unet.attn_processors)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(
                        CNAttnProcessor(num_tokens=self.num_tokens)
                    )
            else:
                self.pipe.controlnet.set_attn_processor(
                    CNAttnProcessor(num_tokens=self.num_tokens)
                )

    def generate(
        self,
        pil_images=None,
        prompt=None,
        negative_prompt=None,
        subjects=None,
        scale=1.0,
        num_samples=1,
        num_inference_steps=30,
        seed=None,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1  # not support multiple prompts

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if subjects:
            self.load_textual_concept(prompt, subjects)
        else:
            raise ValueError("Subjects must be provided")

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image in pil_images:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
                1, num_samples, 1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, *image_prompt_embeds_list], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, *uncond_image_prompt_embeds_list], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(
            clip_image, output_hidden_states=True
        ).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class ConceptrolIPAdapterPlus(ConceptrolIPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(
            clip_image, output_hidden_states=True
        ).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(
            clip_image, output_hidden_states=True
        ).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_images=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=42,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1  # not support multiple prompts

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image in pil_images:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
                1, num_samples, 1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, *image_prompt_embeds_list], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, *uncond_image_prompt_embeds_list], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class ConceptrolIPAdapterPlusXL(ConceptrolIPAdapterXL):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(
            clip_image, output_hidden_states=True
        ).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_images=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        subjects=None,
        num_samples=1,
        seed=42,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1  # not support multiple prompts

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        if subjects:
            self.load_textual_concept(prompt, subjects)
        else:
            raise ValueError("Subjects must be provided")

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds_list = []
        uncond_image_prompt_embeds_list = []
        for pil_image in pil_images:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image
            )
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
                1, num_samples, 1
            )
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
                bs_embed * num_samples, seq_len, -1
            )
            image_prompt_embeds_list.append(image_prompt_embeds)
            uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, *image_prompt_embeds_list], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, *uncond_image_prompt_embeds_list], dim=1
            )

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
