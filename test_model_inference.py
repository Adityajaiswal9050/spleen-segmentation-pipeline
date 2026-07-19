import torch
from monai.networks.nets import UNet

# Same architecture as train_spleen.py / evaluate_spleen.py -- kept in sync manually
# since there's no shared model-definition module (yet).
IN_CHANNELS = 1
OUT_CHANNELS = 2
PATCH_SIZE = (64, 64, 64)


def build_model():
    return UNet(
        spatial_dims=3, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
    )


def test_model_forward_pass_shape_and_dtype():
    model = build_model()
    model.eval()

    dummy_input = torch.rand(1, IN_CHANNELS, *PATCH_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, OUT_CHANNELS, *PATCH_SIZE)
    assert output.dtype == torch.float32


def test_model_output_is_finite():
    model = build_model()
    model.eval()

    dummy_input = torch.rand(1, IN_CHANNELS, *PATCH_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    assert torch.isfinite(output).all()
