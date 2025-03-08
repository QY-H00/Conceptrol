import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

from ip_adapter import (
    ConceptrolIPAdapterPlus,
    ConceptrolIPAdapterPlusXL,
)
from ip_adapter.custom_pipelines import (
    StableDiffusionCustomPipeline,
    StableDiffusionXLCustomPipeline,
)
from omini_control.conceptrol import Conceptrol
from omini_control.flux_conceptrol_pipeline import FluxConceptrolPipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"

title = r"""
<h1 align="center">Conceptrol: Concept Control of Zero-shot Personalized Image Generation</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/QY-H00/Conceptrol/tree/public' target='_blank'><b>Conceptrol: Concept Control of Zero-shot Personalized Image Generation</b></a>.<br>
How to use:<br>
1. Input text prompt, visual specification and the textual concept of the personalized target.
2. Choose your preferrd base model, the first time for switching might take 30 minutes to download the model.
3. For each inference, SD-series takes about 10s, SDXL-series takes about 30s, FLUX takes about 50s.
4. Click the <b>Generate</b> button to enjoy! 😊
"""

article = r"""
---
✒️ **Citation**
<br>
If you found this demo/our paper useful, please consider citing:
```bibtex
@article{he2025conceptrol,
  title={Conceptrol: Concept Control of Zero-shot Personalized Image Generation},
  author={He, Qiyuan and Yao, Angela},
  journal={arXiv preprint arXiv:2403.17924},
  year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue in our <a href='https://github.com/QY-H00/Conceptrol/tree/public' target='_blank'><b>Github Repo</b></a> or directly reach us out at <b>qhe@u.nus.edu.sg</b>.
"""

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = False
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"
PREVIEW_IMAGES = False

# Default settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adapter_name = "h94/IP-Adapter/models/ip-adapter-plus_sd15.bin"
pipe = StableDiffusionCustomPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)
pipeline = ConceptrolIPAdapterPlus(pipe, "", adapter_name, device, num_tokens=16)

def change_model_fn(model_name: str) -> None:
    global device, pipeline
    
    # Clear GPU memory
    if torch.cuda.is_available():
        if pipeline is not None:
            del pipeline
        torch.cuda.empty_cache()
        
    name_mapping = {
        "SD1.5-512": "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        "AOM3 (SD-based)": "hogiahien/aom3",
        "RealVis-v5.1 (SD-based)": "SG161222/Realistic_Vision_V5.1_noVAE",
        "SDXL-1024": "stabilityai/stable-diffusion-xl-base-1.0",
        "RealVisXL-v5.0 (SDXL-based)": "SG161222/RealVisXL_V5.0",
        "Playground-XL-v2 (SDXL-based)": "playgroundai/playground-v2.5-1024px-aesthetic",
        "Animagine-XL-v4.0 (SDXL-based)": "cagliostrolab/animagine-xl-4.0",
        "FLUX-schnell": "black-forest-labs/FLUX.1-schnell"
    }
    if "XL" in model_name:
        adapter_name = "h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
        pipe = StableDiffusionXLCustomPipeline.from_pretrained(
            name_mapping[model_name],
            # variant="fp16",
            torch_dtype=torch.float16,
            feature_extractor=None
        )
        pipeline = ConceptrolIPAdapterPlusXL(pipe, "", adapter_name, device, num_tokens=16)
        globals()["pipeline"] = pipeline
    
    elif "FLUX" in model_name:
        adapter_name = "Yuanshi/OminiControl"
        pipeline = FluxConceptrolPipeline.from_pretrained(
            name_mapping[model_name], torch_dtype=torch.bfloat16
        ).to(device)
        pipeline.load_lora_weights(
            adapter_name,
            weight_name="omini/subject_512.safetensors",
            adapter_name="subject",
        )
        config = {"name": "conceptrol"}
        conceptrol = Conceptrol(config)
        pipeline.load_conceptrol(conceptrol)
        globals()["pipeline"] = pipeline
        globals()["pipeline"].to(device, dtype=torch.bfloat16)
    
    elif "XL" not in model_name and "FLUX" not in model_name:
        adapter_name = "h94/IP-Adapter/models/ip-adapter-plus_sd15.bin"
        pipe = StableDiffusionCustomPipeline.from_pretrained(
            name_mapping[model_name],
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None
        )
        pipeline = ConceptrolIPAdapterPlus(pipe, "", adapter_name, device, num_tokens=16)
        globals()["pipeline"] = pipeline
    else:
        raise KeyError("Not supported model name!")


def save_image(img, index):
    unique_name = f"{index}.png"
    img = Image.fromarray(img)
    img.save(unique_name)
    return unique_name


def get_example() -> list[list[str | float | int]]:
    case = [
        [
            "A statue is reading the book in the cafe, best quality, high quality",
            "statue",
            "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            Image.open("demo/statue.jpg"),
            50,
            6.0,
            1.0,
            0.2,
            42,
            "RealVis-v5.1 (SD-based)"
        ],
        [
            "A hyper-realistic, high-resolution photograph of an astronaut in a meticulously detailed space suit riding a majestic horse across an otherworldly landscape. The image features dynamic lighting, rich textures, and a cinematic atmosphere, capturing every intricate detail in stunning clarity.",
            "horse",
            "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            Image.open("demo/horse.jpg"),
            50,
            6.0,
            1.0,
            0.2,
            42,
            "RealVisXL-v5.0 (SDXL-based)"
        ],
        [
            "A man wearing a T-shirt walking on the street",
            "T-shirt",
            "",
            Image.open("demo/t-shirt.jpg"),
            20,
            3.5,
            1.0,
            0.0,
            42,
            "FLUX-schnell"
        ]
    ]
    return case


def change_generate_button_fn(enable: int) -> gr.Button:
    if enable == 0:
        return gr.Button(interactive=False, value="Switching Model...")
    else:
        return gr.Button(interactive=True, value="Generate")


def dynamic_gallery_fn():
    return gr.Image(label="Result", show_label=False)


@torch.no_grad()
def generate(
    prompt="a statue is reading the book in the cafe",
    subject="cat",
    negative_prompt="",
    image=None,
    num_inference_steps=20,
    guidance_scale=3.5,
    condition_scale=1.0,
    control_guidance_start=0.0,
    seed=0,
    model_name="RealVis-v5.1 (SD-based)"
) -> np.ndarray:
    global pipeline
    change_model_fn(model_name)
    if isinstance(pipeline, FluxConceptrolPipeline):
        images = pipeline(
            prompt=prompt,
            image=image,
            subject=subject,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            condition_scale=condition_scale,
            control_guidance_start=control_guidance_start,
            height=512,
            width=512,
            seed=seed,
        ).images[0]
    elif isinstance(pipeline, ConceptrolIPAdapterPlus) or isinstance(pipeline, ConceptrolIPAdapterPlusXL):
        images = pipeline.generate(
            prompt=prompt,
            pil_images=[image],
            subjects=[subject],
            num_samples=1,
            num_inference_steps=50,
            scale=condition_scale,
            negative_prompt=negative_prompt,
            control_guidance_start=control_guidance_start,
            seed=seed,
        )[0]
    else:
        raise TypeError("Unsupported Pipeline")

    return images

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row(elem_classes="grid-container"):
        with gr.Group():
            with gr.Row(elem_classes="flex-grow"):
                with gr.Column(elem_classes="grid-item"):  # 左侧列
                    prompt = gr.Text(
                        label="Prompt",
                        max_lines=3,
                        placeholder="Enter the Descriptive Prompt",
                        interactive=True,
                        value="A statue is reading the book in the cafe, best quality, high quality",
                    )
                    textual_concept = gr.Text(
                        label="Textual Concept",
                        max_lines=3,
                        placeholder="Enter the Textual Concept required customization",
                        interactive=True,
                        value="statue",
                    )
                    negative_prompt = gr.Text(
                        label="Negative prompt",
                        max_lines=3,
                        placeholder="Enter a Negative Prompt",
                        interactive=True,
                        value="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
                    )
        
        with gr.Row(elem_classes="flex-grow"):
            image_prompt = gr.Image(
                    label="Reference Image for customization",
                    interactive=True,
                    height=280
                )
                

        with gr.Group():
            with gr.Column(elem_classes="grid-item"):  # 右侧列
                with gr.Row(elem_classes="flex-grow"):
                    
                    with gr.Group():
                        # result = gr.Gallery(label="Result", show_label=False, rows=1, columns=1)
                        result = gr.Image(label="Result", show_label=False, height=238, width=256)
                        generate_button = gr.Button(value="Generate", variant="primary")

    with gr.Accordion("Advanced options", open=True):
        with gr.Row():
            with gr.Column():
                # with gr.Row(elem_classes="flex-grow"):
                model_choice = gr.Dropdown(
                    [
                        "AOM3 (SD-based)",
                        "SD1.5-512",
                        "RealVis-v5.1 (SD-based)",
                        "SDXL-1024", 
                        "RealVisXL-v5.0 (SDXL-based)",
                        "Animagine-XL-v4.0 (SDXL-based)",
                        "FLUX-schnell"
                    ],
                    label="Model",
                    value="RealVis-v5.1 (SD-based)",
                    interactive=True,
                    info="XL-Series takes longer time and FLUX takes even more",
                )
                condition_scale = gr.Slider(
                    label="Condition Scale of Reference Image",
                    minimum=0.4,
                    maximum=1.5,
                    step=0.05,
                    value=1.0,
                    interactive=True,
                )
                warmup_ratio = gr.Slider(
                    label="Warmup Ratio",
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    value=0.2,
                    interactive=True,
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0,
                    maximum=10,
                    step=0.1,
                    value=5.0,
                    interactive=True,
                )
        num_inference_steps = gr.Slider(
            label="Inference Steps",
            minimum=10,
            maximum=50,
            step=1,
            value=50,
            interactive=True,
        )
        with gr.Column():
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

    gr.Examples(
        examples=get_example(),
        inputs=[
            prompt,
            textual_concept,
            negative_prompt,
            image_prompt,
            num_inference_steps,
            guidance_scale,
            condition_scale,
            warmup_ratio,
            seed,
            model_choice
        ],
        cache_examples=CACHE_EXAMPLES,
    )

    # model_choice.change(
    #     fn=change_generate_button_fn,
    #     inputs=gr.Number(0, visible=False),
    #     outputs=generate_button,
    # )
    
    # .then(fn=change_model_fn, inputs=model_choice).then(
    #     fn=change_generate_button_fn,
    #     inputs=gr.Number(1, visible=False),
    #     outputs=generate_button,
    # )

    inputs = [
        prompt,
        textual_concept,
        negative_prompt,
        image_prompt,
        num_inference_steps,
        guidance_scale,
        condition_scale,
        warmup_ratio,
        seed,
        model_choice
    ]
    generate_button.click(
        fn=dynamic_gallery_fn,
        outputs=result,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    gr.Markdown(article)

demo.launch()
