import torch
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable
from .transformer import tranformer_forward
from .condition import Condition
from .conceptrol import Conceptrol

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)

denoising_images = []


def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    )


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_scale(pipe, condition_scale):
    for name, module in pipe.transformer.named_modules():
        if not name.endswith(".attn"):
            continue
        module.c_factor = torch.ones(1, 1) * condition_scale


class FluxConceptrolPipeline(FluxPipeline):

    def find_subsequence(self, text, sub):
        sub_len = len(sub)
        for i in range(len(text) - sub_len + 1):
            if text[i : i + sub_len] == sub:
                return i, i + sub_len  # Return start and end indices
        return None

    def locate_subject(self, prompt, subject, max_length=512):
        text_inputs = self.tokenizer_2.tokenize(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        subject_inputs = self.tokenizer_2.tokenize(
            subject,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        print("Text Inputs:", text_inputs)
        print("Sbject Inputs:", subject_inputs)
        print(self.find_subsequence(text_inputs, subject_inputs))
        return self.find_subsequence(text_inputs, subject_inputs)

        text_input_ids = text_inputs
        return (
            text_input_ids.index(subject_inputs[0]),
            text_input_ids.index(subject_inputs[-1]) + 1,
        )

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        model_config: Optional[Dict[str, Any]] = {},
        condition_scale: float = 1.0,
        subject: Optional[str] = None,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        conceptrol: Conceptrol = None,
        seed: int = 42,
        **params: dict,
    ):
        seed_everything(seed)

        conditions = [Condition("subject", image.convert("RGB").resize((512, 512)))]
        if condition_scale != 1:
            for name, module in self.transformer.named_modules():
                if not name.endswith(".attn"):
                    continue
                module.c_factor = torch.ones(1, 1) * condition_scale

        (
            prompt,
            prompt_2,
            height,
            width,
            num_inference_steps,
            timesteps,
            guidance_scale,
            num_images_per_prompt,
            generator,
            latents,
            prompt_embeds,
            pooled_prompt_embeds,
            output_type,
            return_dict,
            joint_attention_kwargs,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        ) = prepare_params(**params)

        if subject is not None:
            textual_concept_idx = self.locate_subject(params["prompt"], subject)
        else:
            raise ValueError("Subject has to be provided")

        if textual_concept_idx is None:
            raise ValueError("Textual concept idx has to be provided")

        conceptrol.register(textual_concept_idx)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 4.1. Prepare conditions
        condition_latents, condition_ids, condition_type_ids = ([] for _ in range(3))
        use_condition = conditions is not None or []
        if use_condition:
            assert len(conditions) <= 1, "Only one condition is supported for now."
            self.set_adapters(conditions[0].condition_type)
            for condition in conditions:
                tokens, ids, type_id = condition.encode(self)
                condition_latents.append(tokens)  # [batch_size, token_n, token_dim]
                condition_ids.append(ids)  # [token_n, id_dim(3)]
                condition_type_ids.append(type_id)  # [token_n, 1]
            condition_latents = torch.cat(condition_latents, dim=1)
            condition_ids = torch.cat(condition_ids, dim=0)
            if condition.condition_type == "subject":
                condition_ids[:, 2] += width // 16
            condition_type_ids = torch.cat(condition_type_ids, dim=0)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if (i / len(timesteps) < control_guidance_start) or (
                    (i + 1) / len(timesteps) > control_guidance_end
                ):
                    set_scale(self, 0.5)  # Warmup required for the first few steps
                else:
                    set_scale(self, condition_scale)
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                noise_pred = tranformer_forward(
                    self.transformer,
                    model_config=model_config,
                    conceptrol=conceptrol,
                    # Inputs of the condition (new feature)
                    condition_latents=condition_latents if use_condition else None,
                    condition_ids=condition_ids if use_condition else None,
                    condition_type_ids=condition_type_ids if use_condition else None,
                    # Inputs to the original transformer
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, latents, callback_kwargs
                    )

                    global denoising_images
                    denoising_images.append(callback_outputs)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if condition_scale != 1:
            for name, module in self.transformer.named_modules():
                if not name.endswith(".attn"):
                    continue
                del module.c_factor

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
