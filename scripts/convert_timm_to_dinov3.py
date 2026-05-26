#!/usr/bin/env python3
"""Convert timm DINOv3 weights to the custom dinov3 DinoVisionTransformer format."""

import argparse
import sys

import torch


def convert_timm_to_dinov3(timm_weights_path: str, output_path: str):
    """Convert timm safetensors weights to custom dinov3 format."""
    import safetensors.torch

    from dinov3.hub.backbones import dinov3_vitb16

    timm_state = safetensors.torch.load_file(timm_weights_path)

    model = dinov3_vitb16(pretrained=False)
    custom_state = model.state_dict()

    key_mapping = {
        "cls_token": "cls_token",
        "reg_token": "storage_tokens",
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }

    num_blocks = 12
    for i in range(num_blocks):
        key_mapping.update({
            f"blocks.{i}.norm1.weight": f"blocks.{i}.norm1.weight",
            f"blocks.{i}.norm1.bias": f"blocks.{i}.norm1.bias",
            f"blocks.{i}.attn.qkv.weight": f"blocks.{i}.attn.qkv.weight",
            f"blocks.{i}.attn.proj.weight": f"blocks.{i}.attn.proj.weight",
            f"blocks.{i}.attn.proj.bias": f"blocks.{i}.attn.proj.bias",
            f"blocks.{i}.gamma_1": f"blocks.{i}.ls1.gamma",
            f"blocks.{i}.norm2.weight": f"blocks.{i}.norm2.weight",
            f"blocks.{i}.norm2.bias": f"blocks.{i}.norm2.bias",
            f"blocks.{i}.mlp.fc1.weight": f"blocks.{i}.mlp.fc1.weight",
            f"blocks.{i}.mlp.fc1.bias": f"blocks.{i}.mlp.fc1.bias",
            f"blocks.{i}.mlp.fc2.weight": f"blocks.{i}.mlp.fc2.weight",
            f"blocks.{i}.mlp.fc2.bias": f"blocks.{i}.mlp.fc2.bias",
            f"blocks.{i}.gamma_2": f"blocks.{i}.ls2.gamma",
        })

    # Copy mapped weights
    num_copied = 0
    for timm_key, custom_key in key_mapping.items():
        if timm_key in timm_state and custom_key in custom_state:
            if timm_state[timm_key].shape == custom_state[custom_key].shape:
                custom_state[custom_key].copy_(timm_state[timm_key])
                num_copied += 1
            else:
                print(f"Shape mismatch: {timm_key} {timm_state[timm_key].shape} vs {custom_key} {custom_state[custom_key].shape}")
        else:
            print(f"Key not found: timm={timm_key in timm_state}, custom={custom_key in custom_state}")

    keys_kept = 0
    for k in custom_state:
        if k not in key_mapping.values():
            keys_kept += 1

    print(f"Copied {num_copied} weight tensors from timm")
    print(f"Kept {keys_kept} custom-only tensors (qkv bias/mask, rope, mask_token)")

    model.load_state_dict(custom_state, strict=True)
    torch.save({"model": model.state_dict()}, output_path)
    print(f"Saved converted weights to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert timm DINOv3 weights to custom dinov3 format")
    parser.add_argument("input", help="Path to timm model.safetensors file")
    parser.add_argument("--output", "-o", default="dinov3_vitb16_converted.pth", help="Output .pth path")
    args = parser.parse_args()

    convert_timm_to_dinov3(args.input, args.output)
