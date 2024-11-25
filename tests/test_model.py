import pytest
import matplotlib.pyplot as plt
from fml.model import *
from fml.utils import *


@pytest.fixture
def model():
    config = MAEConfig()  # Uses default tiny configuration
    model = MAELite(config).to("cuda")
    return model


@pytest.fixture
def pretrained_model(model):
    # todo: autodownload ckpt from gdrive if missing
    load_pretrained_weights(model, "ckpt/mae_tiny_400e.pth.tar")
    return model


@pytest.fixture
def sample_image():
    return prepare_sample_input("test.png")


def test_model_forward_pas(pretrained_model, sample_image):
    pretrained_model.eval()
    with torch.no_grad():
        loss, pred, mask, _ = pretrained_model(sample_image, mask_ratio=0.75)
    assert loss is not None
    assert pred is not None
    assert mask is not None


def test_visual_reconstruction(pretrained_model, sample_image):
    def reconstruct_image(model, image, use_amp=False):
        torch.manual_seed(0)
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                _, pred, _, _ = model(image, mask_ratio=0.75)
            reconstructed = model.unpatchify(pred).cpu().squeeze(0)
            return denorm(torch.clamp(reconstructed, 0, 1))

    pretrained_model.eval()
    sample_image_denorm = denorm(sample_image.cpu().squeeze(0))
    reconstructed_fp32 = reconstruct_image(pretrained_model, sample_image)
    reconstructed_amp = reconstruct_image(pretrained_model, sample_image, use_amp=True)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image_denorm.permute(1, 2, 0))
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_fp32.permute(1, 2, 0))
    plt.title("Reconstructed FP32")
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_amp.permute(1, 2, 0))
    plt.title("Reconstructed AMP")
    plt.savefig("test_visual_reconstruction.png")
    plt.show()
    plt.close()
