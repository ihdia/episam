# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default":         build_sam_vit_h,
    "vit_h":           build_sam_vit_h,
    "vit_l":           build_sam_vit_l,
    "vit_b":           build_sam_vit_b,
}


def _make_sam(encoder_embed_dim, encoder_depth, encoder_num_heads,
              encoder_global_attn_indexes, num_multimask_outputs=3, num_extra_tokens=1):
    """
    Construct a SAM model with the given architecture parameters.

    Args:
        num_multimask_outputs: Number of mask tokens (default 3 for standard SAM).
                               The total number of output tokens will be
                               1 (iou_token) + num_multimask_outputs + 2 (left/right neighbors).
    """
    prompt_embed_dim       = 256
    image_size             = 1024
    vit_patch_size         = 16
    image_embedding_size   = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_multimask_outputs,
            num_extra_tokens=num_extra_tokens,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    return sam



def _load_standard_checkpoint(sam, checkpoint):
    """
    Load a checkpoint into a standard 4-token SAM model.

    Handles:
      - Vanilla pretrained SAM (3 or 4 tokens)
      - Training checkpoints with nested 'model_state_dict'
      - Loads with strict=False to avoid key mismatches
    """
    print(f"Loading checkpoint: {checkpoint}")
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location="cpu", weights_only=False)

    # Unwrap training checkpoint if needed
    if 'model_state_dict' in state_dict:
        print("  → Detected training checkpoint format (nested 'model_state_dict')")
        state_dict = state_dict['model_state_dict']

    old_tokens = state_dict['mask_decoder.mask_tokens.weight']
    n = old_tokens.shape[0]
    print(f"  → Checkpoint has {n} mask tokens")

    if n == 3:
        print("  → Standard SAM checkpoint (3 tokens) – loading directly.")
    elif n == 4:
        print("  → 4-token checkpoint – loading directly.")
    else:
        raise ValueError(
            f"_load_standard_checkpoint expects 3 or 4 tokens, got {n}. "
            f"Use _load_neighbor_aware_checkpoint for 6+ token models.")

    missing, unexpected = sam.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  → Missing keys (randomly initialised): {missing}")
    if unexpected:
        print(f"  → Unexpected keys (ignored): {unexpected}")
    print("  ✓ Checkpoint loaded successfully.\n")


def _build_sam_standard(encoder_embed_dim, encoder_depth, encoder_num_heads,
                        encoder_global_attn_indexes, checkpoint=None):
    """
    Build a SAM model with a STANDARD 4-token decoder (1 iou + 3 mask tokens).

    This is the correct architecture for line-segmentation fine-tuning where
    multimask_output=False routes all gradients through token 0.

    No neighbor tokens.
    """
    # Build with standard 3 multimask outputs (+ 1 iou = 4 total)
    sam = _make_sam(
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_global_attn_indexes=encoder_global_attn_indexes,
        num_multimask_outputs=3,
        num_extra_tokens=1
    )

    if checkpoint is not None:
        _load_standard_checkpoint(sam, checkpoint)

    return sam



def _load_neighbor_aware_checkpoint(sam, checkpoint, target_num_tokens=6):
    """
    Load a checkpoint into the 6-token neighbor-aware SAM model.

    Handles three cases:
      1. Pretrained SAM (4 tokens) → Expand to 6
      2. Already fine-tuned 6-token model → Load as-is
      3. Old 8-token model → Trim to 6 (keep tokens 0-5, discard 6-7)

    Token breakdown for 6-token model:
      Token 0: Primary mask
      Token 1: Ambiguity mask A
      Token 2: Ambiguity mask B
      Token 3: Left neighbor mask
      Token 4: Right neighbor mask
      Token 5: (unused but allocated)
    """
    print(f"Loading checkpoint: {checkpoint}")
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location="cpu", weights_only=False)

    # Unwrap training checkpoint if needed
    if 'model_state_dict' in state_dict:
        print("  → Detected training checkpoint format (nested 'model_state_dict')")
        state_dict = state_dict['model_state_dict']

    old_tokens = state_dict['mask_decoder.mask_tokens.weight']
    n = old_tokens.shape[0]
    print(f"  → Checkpoint has {n} mask tokens, target is {target_num_tokens}")

    # ────────────────────────────────────────────────────────────────────────
    # Case 1: Pretrained SAM (4 tokens) → Expand to 6
    # ────────────────────────────────────────────────────────────────────────
    if n == 4:
        print("  → Expanding from 4 to 6 tokens (pretrained SAM)")
        print("     Tokens 0-3: Copied from checkpoint")
        print("     Tokens 4-5: Initialized from tokens 1-2 (neighbor tokens)")

        # Copy tokens 1-2 as the new neighbor tokens (4-5)
        new_neighbor_tokens = old_tokens[1:3].clone()
        new_mask_tokens = torch.cat([old_tokens, new_neighbor_tokens], dim=0)  # [6, 256]
        state_dict['mask_decoder.mask_tokens.weight'] = new_mask_tokens

        # Expand IoU head: [4, 256] → [6, 256] and [4] → [6]
        old_iou_weight = state_dict['mask_decoder.iou_prediction_head.layers.2.weight']
        old_iou_bias   = state_dict['mask_decoder.iou_prediction_head.layers.2.bias']

        new_iou_weight = torch.cat([
            old_iou_weight,
            torch.zeros(2, old_iou_weight.shape[1],
                       device=old_iou_weight.device, dtype=old_iou_weight.dtype)
        ], dim=0)  # [6, 256]

        new_iou_bias = torch.cat([
            old_iou_bias,
            torch.zeros(2, device=old_iou_bias.device, dtype=old_iou_bias.dtype)
        ], dim=0)  # [6]

        state_dict['mask_decoder.iou_prediction_head.layers.2.weight'] = new_iou_weight
        state_dict['mask_decoder.iou_prediction_head.layers.2.bias']   = new_iou_bias

    # ────────────────────────────────────────────────────────────────────────
    # Case 2: Already fine-tuned 6-token model → Use as-is
    # ────────────────────────────────────────────────────────────────────────
    elif n == 6:
        print("  → Checkpoint already has 6 tokens – loading directly.")

    # ────────────────────────────────────────────────────────────────────────
    # Case 3: Old 8-token model → Trim to 6
    # ────────────────────────────────────────────────────────────────────────
    elif n == 8:
        print("  → Old 8-token checkpoint detected – trimming to 6 tokens.")
        print("     Keeping tokens 0-5, discarding tokens 6-7")

        trimmed_tokens = old_tokens[:6].clone()
        state_dict['mask_decoder.mask_tokens.weight'] = trimmed_tokens

        # Trim IoU head: [8, 256] → [6, 256] and [8] → [6]
        old_iou_weight = state_dict['mask_decoder.iou_prediction_head.layers.2.weight']
        old_iou_bias   = state_dict['mask_decoder.iou_prediction_head.layers.2.bias']

        state_dict['mask_decoder.iou_prediction_head.layers.2.weight'] = old_iou_weight[:6]
        state_dict['mask_decoder.iou_prediction_head.layers.2.bias']   = old_iou_bias[:6]

    else:
        raise ValueError(
            f"_load_neighbor_aware_checkpoint expects 4, 6, or 8 tokens, got {n}. "
            f"Unsupported checkpoint format.")

    # Load with strict=False to handle any minor mismatches
    print(f"  → Final mask_tokens shape: {state_dict['mask_decoder.mask_tokens.weight'].shape}")
    missing, unexpected = sam.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  → Missing keys (randomly initialised): {missing}")
    if unexpected:
        print(f"  → Unexpected keys (ignored): {unexpected}")
    print("  ✓ Checkpoint loaded successfully.\n")


def _build_sam(encoder_embed_dim, encoder_depth, encoder_num_heads,
               encoder_global_attn_indexes, checkpoint=None):
    """
    Build the 6-token neighbor-aware SAM model.

    Architecture:
      - 1 IoU prediction token
      - 3 mask tokens (primary + 2 ambiguity masks)
      - 2 neighbor tokens (left + right)
      = 6 total output tokens

    Used by train_centroid_plus_bbox_prompt.py and other character-level models.
    """
    # Build with 3 multimask outputs (+ 1 iou + 2 neighbors = 6 total)
    sam = _make_sam(
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_global_attn_indexes=encoder_global_attn_indexes,
        num_multimask_outputs=3,
        num_extra_tokens=3
    )

    if checkpoint is not None:
        _load_neighbor_aware_checkpoint(sam, checkpoint, target_num_tokens=6)

    return sam