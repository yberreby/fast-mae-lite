"""
Demo inference script - cobbled together.
"""

import matplotlib

matplotlib.use("Qt5Agg")
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fml.model import MAELite, MAEConfig
from fml.utils import prepare_sample_input, denorm


def load_checkpoint(model: MAELite, path: str, device: torch.device) -> None:
    """Load checkpoint, handling both legacy and training script formats."""
    ckpt = torch.load(path, map_location=device)

    # If this is a training checkpoint
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        # Remove _orig_mod prefix from compiled model
        if all(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    # If this is a raw state dict
    elif all(k.startswith(("encoder.", "decoder.")) for k in ckpt.keys()):
        model.load_state_dict(ckpt)
    # If this is a legacy checkpoint
    else:
        model.load_legacy_weights(path)


def main():
    parser = argparse.ArgumentParser(description="MAE finetuned inference demo.")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to finetuned .pt checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build a default config and model
    config = MAEConfig(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        decoder_embed_dim=96,
    )
    model = MAELite(config, device).to(device)

    # 2. Load the checkpoint with improved loader
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    # 3. Load and prepare the image
    img = prepare_sample_input(args.image, device=device)

    # 4. Forward pass to get reconstruction and the patch mask
    with torch.no_grad():
        loss, pred, mask, ids_shuffle = model(img, mask_ratio=args.mask_ratio)
        reconstruction = model.unpatchify(pred)

    # 5. Build the masked-out input for display
    B, _, H, W = img.shape
    h = w = H // model.config.patch_size
    mask_reshaped = mask.reshape(B, h, w).unsqueeze(1)
    mask_upsampled = F.interpolate(
        mask_reshaped.float(), scale_factor=model.config.patch_size, mode="nearest"
    )
    masked_input = img * (1 - mask_upsampled)

    # 6. Denormalize & convert to CPU numpy for plotting
    img_denorm = denorm(img[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    masked_denorm = denorm(masked_input[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    recon_denorm = denorm(reconstruction[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()

    # 7. Visualize side by side
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_denorm)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(masked_denorm)
    plt.title("Masked Input")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(recon_denorm)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
