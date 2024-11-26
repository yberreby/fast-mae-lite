from PIL import Image
import torch
import torchvision.transforms as transforms


fixed_mean = [0.485, 0.456, 0.406]
fixed_std = [0.229, 0.224, 0.225]


def prepare_sample_input(image_path: str, device: str = "cuda") -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=fixed_mean, std=fixed_std),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image


def denorm(tensor: torch.Tensor, mean=fixed_mean, std=fixed_std) -> torch.Tensor:
    mean = torch.tensor(mean).to(tensor.device).view(3, 1, 1)
    std = torch.tensor(std).to(tensor.device).view(3, 1, 1)
    return tensor * std + mean
