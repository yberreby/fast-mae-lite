import re

def _remap_attention_weights(state_dict: dict, block_path: str) -> dict:
    """
    Remap custom attention weights to PyTorch MultiheadAttention format.
    """
    new_state_dict = {}

    # Get original weights using full paths
    qkv_weight = state_dict[f"{block_path}attn.qkv.weight"]
    qkv_bias = state_dict[f"{block_path}attn.qkv.bias"]
    proj_weight = state_dict[f"{block_path}attn.proj.weight"]
    proj_bias = state_dict[f"{block_path}attn.proj.bias"]

    # Transform the block path to new format
    new_prefix = block_path.replace("module.model.", "")
    if new_prefix.startswith("blocks."):
        new_prefix = f"encoder.{new_prefix}"
    elif new_prefix.startswith("decoder_blocks."):
        new_prefix = new_prefix.replace("decoder_blocks.", "decoder.decoder_blocks.")

    # Store with new keys matching the mha attribute name
    new_state_dict[f"{new_prefix}attn.mha.in_proj_weight"] = qkv_weight
    new_state_dict[f"{new_prefix}attn.mha.in_proj_bias"] = qkv_bias
    new_state_dict[f"{new_prefix}attn.mha.out_proj.weight"] = proj_weight
    new_state_dict[f"{new_prefix}attn.mha.out_proj.bias"] = proj_bias

    return new_state_dict


def remap_legacy_state_dict_to_new_format(state_dict):
    new_state_dict = {}

    # Handle attention blocks first
    for k in state_dict.keys():
        if "attn.qkv.weight" in k:
            # Extract block path by removing the attention component
            block_path = k.rsplit("attn.qkv.weight", 1)[0]
            attention_dict = _remap_attention_weights(state_dict, block_path)
            new_state_dict.update(attention_dict)

    # Handle all non-attention weights
    key_transforms = {
        r"^module\.model\.patch_embed": "encoder.patch_embed",
        r"^module\.model\.cls_token": "encoder.cls_token",
        r"^module\.model\.pos_embed": "encoder.pos_embed",
        r"^module\.model\.blocks\.": "encoder.blocks.",
        r"^module\.model\.norm\.": "encoder.norm.",
        r"^module\.model\.decoder_embed\.": "decoder.decoder_embed.",
        r"^module\.model\.decoder_pos_embed": "decoder.decoder_pos_embed",
        r"^module\.model\.decoder_blocks\.": "decoder.decoder_blocks.",
        r"^module\.model\.decoder_norm\.": "decoder.decoder_norm.",
        r"^module\.model\.decoder_pred\.": "decoder.decoder_pred.",
        r"^module\.model\.mask_token": "decoder.mask_token",
    }

    for k, v in state_dict.items():
        # Skip attention weights as they're handled separately
        if "attn." in k:
            continue

        new_k = k
        for pattern, replacement in key_transforms.items():
            new_k = re.sub(pattern, replacement, new_k)
        new_state_dict[new_k] = v

    return new_state_dict
