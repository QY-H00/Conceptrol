import os
import logging
import torch
import argparse
from datetime import datetime
from PIL import Image, ImageDraw

from omini_control.conceptrol import Conceptrol
from omini_control.flux_conceptrol_pipeline import FluxConceptrolPipeline


def main(
    prompt="A statue is reading the book in the library",
    image_path="demo/book.jpg",
    subject="book",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("temp", exist_ok=True)
    logging.basicConfig(
        filename=f"temp/demo_run_{timestamp}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    logging.info("Loading pipeline and models")
    pipeline = FluxConceptrolPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name="omini/subject_512.safetensors",
        adapter_name="subject",
    )
    logging.info("Pipeline and models loaded successfully")

    seed = 42

    logging.info(f"Prompt: {prompt}; Subject: {subject}")
    image = Image.open(image_path)
    logging.info("Images loaded successfully")

    configs = [{"name": "conceptrol"}, {"name": "ominicontrol"}]
    scale = 1.0
    logging.info("Configurations initialized")

    # Create a row for each config with title
    total_width = 512 * (len(configs) + 1)  # +1 for input image
    row_height = 512
    title_height = 30

    # Create title text
    title = f"{prompt} (Concept: {subject})"

    # Create new image for the row with space for title
    row_image = Image.new("RGB", (total_width, row_height + title_height))

    # Add title text
    draw = ImageDraw.Draw(row_image)
    draw.text((10, 5), title, fill="white")

    # Add input image on left
    row_image.paste(image.resize((512, 512)), (0, title_height))

    # Generate and add result images for each config
    for idx, config in enumerate(configs):
        logging.info(f"Starting {config['name']} with scale {scale} for {subject}")

        conceptrol = Conceptrol(config)
        generated_img = pipeline(
            prompt=prompt,
            image=image,
            subject=subject,
            conceptrol=conceptrol,
            num_inference_steps=20,
            guidance_scale=3.5,
            condition_scale=scale,
            height=512,
            width=512,
            seed=seed,
        ).images[0]

        # Add generated image next to previous image
        x_pos = 512 * (
            idx + 1
        )  # Position after input image and previous generated images
        row_image.paste(generated_img, (x_pos, title_height))
        logging.info(f"Added {config['name']} result to row")

    # Save the complete row with all results
    row_image.save(f"temp/output_{subject}.png")
    logging.info("Saved image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using FluxConceptrolPipeline"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A statue is reading the book in the library",
        help="The prompt text",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="demo/book.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--subject", type=str, default="book", help="The subject concept"
    )

    args = parser.parse_args()
    main(args.prompt, args.image_path, args.subject)
