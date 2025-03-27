import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# global variable
raw_attn_maps = {}
raw_ip_attn_maps = {}
attn_maps = {}
ip_attn_maps = {}


def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            if name not in raw_attn_maps:
                raw_attn_maps[name] = []
            if name not in raw_ip_attn_maps:
                raw_ip_attn_maps[name] = []
            raw_attn_maps[name].append(module.processor.attn_map)
            raw_ip_attn_maps[name].append(module.processor.ip_attn_map)
            del module.processor.attn_map
            del module.processor.ip_attn_map

    return forward_hook


def post_process_attn_maps():
    global raw_attn_maps, raw_ip_attn_maps, attn_maps, ip_attn_maps
    attn_maps = [
        dict(zip(raw_attn_maps.keys(), values))
        for values in zip(*raw_attn_maps.values())
    ]
    ip_attn_maps = [
        dict(zip(raw_ip_attn_maps.keys(), values))
        for values in zip(*raw_ip_attn_maps.values())
    ]

    return attn_maps, ip_attn_maps


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split(".")[-1].startswith("attn2"):
            module.register_forward_hook(hook_fn(name))

    return unet


def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1, 0)
    temp_size = None

    for i in range(0, 5):
        scale = 2**i
        if (target_size[0] // scale) * (target_size[1] // scale) == attn_map.shape[
            1
        ] * 64:
            temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map


def get_net_attn_map(
    image_size, batch_size=2, instance_or_negative=False, detach=True, step=-1
):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []
    net_ip_attn_maps = []

    for _, attn_map in attn_maps[step].items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[
            idx
        ].squeeze()  # get the attention map of text
        attn_map = upscale(attn_map, image_size)
        net_attn_maps.append(attn_map)

    net_attn_maps = torch.mean(torch.stack(net_attn_maps, dim=0), dim=0)

    for _, attn_map in ip_attn_maps[step].items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[
            idx
        ].squeeze()  # get the attention map of text
        attn_map = upscale(attn_map, image_size)
        net_ip_attn_maps.append(attn_map)

    net_ip_attn_maps = torch.mean(torch.stack(net_ip_attn_maps, dim=0), dim=0)

    return net_attn_maps, net_ip_attn_maps


def attnmaps2images(net_attn_maps):
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        normalized_attn_map = (
            (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        )
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        image = Image.fromarray(normalized_attn_map)
        images.append(image)

    return images


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [
                torch.Generator(device).manual_seed(seed_item) for seed_item in seed
            ]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
