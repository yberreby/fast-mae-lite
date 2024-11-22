import pytest
import matplotlib.pyplot as plt
from fml.model import *
from fml.utils import *

@pytest.fixture
def model():
    config = MAEConfig()  # Uses default tiny configuration
    model = MAELite(config).to('cuda')
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
    pretrained_model.eval()
    with torch.no_grad():
        loss, pred, mask, _ = pretrained_model(sample_image, mask_ratio=0.75)

    # Unpatchify the predicted patches to reconstruct the image
    reconstructed = pretrained_model.unpatchify(pred)
    reconstructed = reconstructed.cpu().squeeze(0)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    # Denormalize the images
    sample_image_denorm = denorm(sample_image.cpu().squeeze(0))

    # Outputs are normalized with local statistics. Denormalization using ImageNet
    # doesn't make as much sense.
    # UNMASKED patches will not get correctly reconstructed, that is fine. You can put them back manually or finetune.
    reconstructed = denorm(reconstructed)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image_denorm.permute(1, 2, 0))
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.permute(1, 2, 0))
    plt.title("Reconstructed")
    plt.savefig('test_visual_reconstruction.png')
    plt.show()
    plt.close()
