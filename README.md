<div align="center">
<h1>Conceptrol: Concept Control of Zero-shot Personalized Image Generation</h1>

<div align="center">
<div>
  <a href="https://qy-h00.github.io" target="_blank">He Qiyuan</a>,&nbsp;
  <a href="https://www.comp.nus.edu.sg/~ayao//" target="_blank">Angela Yao</a>
  </sup></a>
  <br>
</div>
<div>
  <a herf=https://cvml.comp.nus.edu.sg>CVML Lab, National University of Singapore</a>
</div>
</br>
</div>

<div align="center">
<a href=https://arxiv.org/abs/2403.17924 target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv height=25px></a>
<a href=https://huggingface.co/spaces/qyoo/Conceptrol target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20Space-276cb4.svg height=25px></a>
<a href=https://qy-h00.github.io/Conceptrol target="_blank"><img src= https://img.shields.io/badge/GitHub%20Project%20Page-bb8a2e.svg?logo=github height=25px></a>
</div>

---

<strong>We propose Conceptrol, a training-free method that boosts zero-shot personalized image generation across <span style="color: #DB4437;">Stable Diffusion</span> / <span style="color: #4285F4;">SDXL</span> / <span style="color: #0F9D58;">FLUX</span> without additional training / data / models.</strong>

<p align="center">
  <img src="demo/teaser.png">
</p>

</div>

## 📌 Release

[03/2025] Code and paper are publicly available.

## 📑 Abstract

<b>TL;DR: <font color="red">Conceptrol</font> </b> is a free lunch that elicits the personalized ability of zero-shot adapter by transforming image condition to visual specification contrained by textual concept, even outperforming fine-tuning methods.

<details><summary>CLICK for the full abstract</summary>
Personalized image generation with text-to-image diffusion models generates unseen images based on reference image content. Zero-shot adapter methods such as IP-Adapter and OminiControl are especially interesting because they do not require test-time fine-tuning.  However, they struggle to balance preserving personalized content and adherence to the text prompt. We identify a critical design flaw resulting in this performance gap: current adapters inadequately integrate personalization images with the textual descriptions. The generated images, therefore, replicate the personalized content rather than adhere to the text prompt instructions. Yet the base text-to-image has strong conceptual understanding capabilities that can be leveraged.

We propose Conceptrol, a simple yet effective framework that enhances zero-shot adapters without adding computational overhead. Conceptrol constrains the attention of visual specification with a textual concept mask that improves subject-driven generation capabilities. It achieves as much as 89\% improvement on personalization benchmarks over the vanilla IP-Adapter and can even outperform fine-tuning approaches such as Dreambooth LoRA.  The source code will be released upon acceptance.
</details>

## 🚗 Quick Start

#### 1. Environment Setup

``` bash
conda create -n conceptrol python=3.10
conda activate conceptrol
pip install -r requirements.txt
```

#### 2. Go to `demo_sd.ipynb` / `demo_sdxl.ipynb` / `demo_flux.py` for fun!

## 🛳️ Local Setup using Gradio

#### 1. Start Gradio Interface
``` bash
pip install gradio
gradio gradio_src/app.py
```

#### 2. Use the GUI for fun!

## 📝 Supporting Models

| Model Name            |  Link                                             |
|-----------------------|-------------------------------------------------------------|
| Stable Diffusion 1.5  | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)   |
| Realistic Vision V5.1 | [SG161222/Realistic_Vision_V5.1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE) |
| Stable Diffusion XL-1024   | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| Animagine XL v4.0 |   [cagliostrolab/animagine-xl-4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0)|
| Realistic Vision XL V5.0 | [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0) |
| FLUX-schnell | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

| Adapter Name            |  Link                                             |
|-----------------------|-------------------------------------------------------------|
| IP-Adapter  | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main)  |
| OminiControl | [Yuanshi/OminiControl](https://huggingface.co/Yuanshi/OminiControl) |

## ✒️Citation

If you found this repository/our paper useful, please consider citing:

``` bibtex
@article{he2024aid,
  title={Conceptrol: Concept Control of Zero-shot Personalized Image Generation},
  author={He, Qiyuan and Yao, Angela},
  journal={tbd},
  year={2024}
}
```


## ❤️ Acknowledgement

We thank the following repositories for their great work: 

[diffusers](https://github.com/huggingface/diffusers), 
[transformers](https://github.com/huggingface/transformers), 
[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), 
[OminiControl](https://github.com/Yuanshi9815/OminiControl)



