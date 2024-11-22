"""
Utilities for loading and verifying MAE Lite models.
"""

import torch
import torchvision.transforms as transforms
from typing import Dict, Tuple, Any


def remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap keys from original checkpoint format to new format.
    """
    new_state_dict = {}

    # Remove potential DDP prefix
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    for key, value in state_dict.items():
        # Handle decoder prefix differences
        if key.startswith('decoder.'):
            key = 'decoder_' + key[8:]

        # Handle potential differences in position embedding names
        if 'pos_embed' in key:
            if not key.startswith('decoder_'):
                key = key.replace('pos_embed', 'pos_embed')

        new_state_dict[key] = value

    return new_state_dict


def validate_state_dict(model_state: Dict[str, torch.Tensor],
                       checkpoint_state: Dict[str, torch.Tensor]) -> Tuple[set, set]:
    """
    Validate checkpoint state against model state.
    Returns sets of missing and unexpected keys.
    """
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint_state.keys())

    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    return missing_keys, unexpected_keys


def load_pretrained_weights(model: torch.nn.Module,
                          checkpoint_path: str,
                          strict: bool = False) -> None:
    """
    Load pretrained weights with validation and proper key mapping.

    Args:
        model: MAE Lite model instance
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce all keys match

    Raises:
        ValueError: If checkpoint is invalid or missing required keys
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Error loading checkpoint: {e}")

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remap keys to match new format
    state_dict = remap_state_dict(state_dict)

    # Validate state dict
    model_state = model.state_dict()
    missing_keys, unexpected_keys = validate_state_dict(model_state, state_dict)

    if strict and (missing_keys or unexpected_keys):
        raise ValueError(
            f"Strict loading failed:\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    # Load weights
    model.load_state_dict(state_dict, strict=not bool(missing_keys))

    return model

def verify_mae_lite(model: torch.nn.Module,
                   sample_image: torch.Tensor,
                   mask_ratio: float = 0.75) -> Tuple[float, torch.Tensor]:
    """
    Verify MAE Lite model is working correctly.

    Args:
        model: MAE Lite model instance
        sample_image: Sample image tensor of shape (B, C, H, W)
        mask_ratio: Ratio of patches to mask

    Returns:
        Tuple of (loss value, reconstructed image)
    """
    model.eval()
    with torch.no_grad():
        # Run forward pass
        loss, pred, mask, _ = model(sample_image, mask_ratio=mask_ratio)

        # Verify shapes
        B = sample_image.shape[0]
        num_patches = model.patch_embed.num_patches
        patch_size = model.config.patch_size

        assert pred.shape == (B, num_patches, patch_size**2 * 3), \
            f"Prediction shape mismatch: got {pred.shape}"
        assert mask.shape == (B, num_patches), \
            f"Mask shape mismatch: got {mask.shape}"

        # Reconstruct image
        pred_img = model.unpatchify(pred)
        assert pred_img.shape == sample_image.shape, \
            f"Reconstructed image shape mismatch: got {pred_img.shape}, expected {sample_image.shape}"

        return loss.item(), pred_img


def prepare_sample_input(image_path: str, device: str = 'cuda') -> torch.Tensor:
    """
    Prepare a sample image for MAE Lite verification.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image


# Usage example
def main():
    """Example usage of MAE Lite model with pretrained weights."""
    # Create model
    config = MAEConfig()
    model = MAELite(config)

    # Load pretrained weights
    checkpoint_path = "path/to/mae_tiny_distill_400e.pth.tar"
    model = load_pretrained_weights(model, checkpoint_path)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Prepare sample input
    sample_image = prepare_sample_input("path/to/sample_image.jpg", device)

    # Verify model
    loss, reconstructed = verify_mae_lite(model, sample_image)
    print(f"Verification loss: {loss:.4f}")

    # Optional: visualize reconstruction
    reconstructed = reconstructed.cpu().squeeze(0)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image.cpu().squeeze(0).permute(1, 2, 0))
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.permute(1, 2, 0))
    plt.title("Reconstructed")
    plt.show()


if __name__ == "__main__":
    main()
