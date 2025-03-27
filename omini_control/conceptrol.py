import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


class Conceptrol:
    def __init__(self, config):
        if "name" not in config:
            raise KeyError("name has to be provided as 'conceptrol' or 'ominicontrol'")

        name = config["name"]
        if name not in ["conceptrol", "ominicontrol"]:
            raise ValueError(
                f"Name must be one of ['conceptrol', 'ominicontrol'], got {name}"
            )

        try:
            log_attn_map = config["log_attn_map"]
        except KeyError:
            log_attn_map = False

        # static
        self.NUM_BLOCKS = 19  # this is fixed for FLUX
        self.M = 512  # num of text tokens, fixed for FLUX
        self.N = 1024  # num of latent / image condtion tokens, fixed for FLUX
        self.EP = -10e6
        self.CONCEPT_BLOCK_IDX = 18

        # fixed during one generation
        self.name = name

        # variable during one generation
        self.textual_concept_mask = None
        self.forward_count = 0

        # log out for visualization
        if log_attn_map:
            self.attn_maps = {"latent_to_concept": [], "latent_to_image": []}

    def __call__(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        attention_mask: torch.Tensor,
        c_factor: float = 1.0,
    ) -> torch.Tensor:

        if not hasattr(self, "textual_concept_idx"):
            raise AttributeError(
                "textual_concept_idx must be registered before calling Conceptrol"
            )

        # Skip computation for ominicontrol
        if self.name == "ominicontrol":
            scale_factor = 1 / math.sqrt(query.size(-1))
            attention_weight = (
                query @ key.transpose(-2, -1) * scale_factor + attention_mask
            )
            attention_probs = torch.softmax(
                attention_weight, dim=-1
            )  # [B, H, M+2N, M+2N]
            return attention_probs

        if not self.textual_concept_idx[0] < self.textual_concept_idx[1]:
            raise ValueError(
                f"register_idx[0] must be less than register_idx[1], "
                f"got {self.textual_concept_idx[0]} >= {self.textual_concept_idx[1]}"
            )

        ### Reset attention mask predefined in ominicontrol
        attention_mask = torch.zeros_like(attention_mask)
        bias = torch.log(c_factor[0])
        # attention of image condition to latent
        attention_mask[-self.N :, self.M : -self.N] = bias
        # attention of latent to image condition
        attention_mask[self.M : -self.N, -self.N :] = bias

        # attention of textual concept to image condition
        attention_mask[
            self.textual_concept_idx[0] : self.textual_concept_idx[1], -self.N :
        ] = bias
        # attention of other words to image condition (set as negative inf)
        attention_mask[: self.textual_concept_idx[0], -self.N :] = self.EP
        attention_mask[self.textual_concept_idx[1] : self.M, -self.N :] = self.EP

        # If there is no textual_concept_mask, it means currently in layers previous to the first concept-specific block
        if self.textual_concept_mask is None:
            self.textual_concept_mask = (
                torch.zeros_like(attention_mask).unsqueeze(0).unsqueeze(0)
            )

        ### Compute attention
        scale_factor = 1 / math.sqrt(query.size(-1))
        attention_weight = (
            query @ key.transpose(-2, -1) * scale_factor
            + attention_mask
            + self.textual_concept_mask
        )
        # [B, H, M+2N, M+2N]
        attention_probs = torch.softmax(attention_weight, dim=-1)

        ### Extract textual concept mask if it's concept-specific block
        is_concept_block = (
            self.forward_count % self.NUM_BLOCKS == self.CONCEPT_BLOCK_IDX
        )
        if is_concept_block:
            # Shape: [B, H, N, S], where S is the token numbers of the subject
            textual_concept_mask_local = attention_probs[
                :,
                :,
                self.M : -self.N,
                self.textual_concept_idx[0] : self.textual_concept_idx[1],
            ]
            # Consider the ratio within context of text
            textual_concept_mask_local = textual_concept_mask_local / torch.sum(
                attention_probs[:, :, self.M : -self.N, : self.M], dim=-1, keepdim=True
            )
            # Average over words and head, Shape: [B, 1, N, 1]
            textual_concept_mask_local = torch.mean(
                textual_concept_mask_local, dim=(-1, 1), keepdim=True
            )
            # Normalize to average as 1
            textual_concept_mask_local = textual_concept_mask_local / torch.mean(
                textual_concept_mask_local, dim=-2, keepdim=True
            )

            self.textual_concept_mask = (
                torch.zeros_like(attention_mask).unsqueeze(0).unsqueeze(0)
            )
            # log(A) in the paper
            self.textual_concept_mask[:, :, self.M : -self.N, -self.N :] = torch.log(
                textual_concept_mask_local
            )

        self.forward_count += 1

        return attention_probs

    def register(self, textual_concept_idx):
        self.textual_concept_idx = textual_concept_idx

    def visualize_attn_map(self, config_name: str, subject: str):
        global global_concept_mask
        global forward_count

        save_dir = f"attn_maps/{config_name}/{subject}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for attn_map_name, attn_maps in self.attn_maps.items():
            if "token_to_token" in attn_map_name:
                continue
            plt.figure()

            rows, cols = 8, 19
            fig, axes = plt.subplots(
                rows, cols, figsize=(64 * cols / 100, 64 * rows / 100)
            )
            fig.subplots_adjust(
                wspace=0.1, hspace=0.1
            )  # Adjust spacing between subplots

            # Plot each array in the list on the grid
            for i, ax in enumerate(axes.flatten()):
                if i < len(attn_maps):  # Only plot existing arrays
                    attn_map = attn_maps[i] / np.amax(attn_maps[i])
                    ax.imshow(attn_map, cmap="viridis")
                    ax.axis("off")  # Turn off axes for clarity
                else:
                    ax.axis("off")  # Turn off unused subplots

            fig.set_size_inches(64 * cols / 100, 64 * rows / 100)
            save_path = os.path.join(save_dir, f"{attn_map_name}.jpg")
            plt.savefig(save_path)
            plt.close()

        for attn_map_name, attn_maps in self.attn_maps.items():
            if "token_to_token" not in attn_map_name:
                continue
            plt.figure()

            rows, cols = 8, 19
            fig, axes = plt.subplots(
                rows, cols, figsize=(2560 * cols / 100, 2560 * rows / 100)
            )
            fig.subplots_adjust(
                wspace=0.1, hspace=0.1
            )  # Adjust spacing between subplots

            # Plot each array in the list on the grid
            for i, ax in enumerate(axes.flatten()):
                if i < len(attn_maps):  # Only plot existing arrays
                    attn_map = attn_maps[i] / np.amax(attn_maps[i])
                    ax.imshow(attn_map, cmap="viridis")
                    ax.axis("off")  # Turn off axes for clarity
                else:
                    ax.axis("off")  # Turn off unused subplots

            fig.set_size_inches(64 * cols / 100, 64 * rows / 100)
            save_path = os.path.join(save_dir, f"{attn_map_name}.jpg")
            plt.savefig(save_path)
            plt.close()

        for attn_map_name in self.attn_maps.keys():
            self.attn_maps[attn_map_name] = []
        global_concept_mask = None
        forward_count = 0
