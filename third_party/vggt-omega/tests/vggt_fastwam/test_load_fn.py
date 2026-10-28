import numpy as np
import pytest
import torch
from PIL import Image

from vggt_omega.utils.load_fn import load_and_preprocess_images


def test_tensor_input_matches_path_api(tmp_path):
    array = np.arange(48 * 96 * 3, dtype=np.uint8).reshape(48, 96, 3)
    path = tmp_path / "image.png"
    Image.fromarray(array).save(path)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)

    from_path = load_and_preprocess_images([path], image_resolution=256)
    from_tensor = load_and_preprocess_images([tensor], image_resolution=256)

    torch.testing.assert_close(from_tensor, from_path)


@pytest.mark.parametrize("resolution", [0, -16, 250])
def test_rejects_invalid_image_resolution_for_tensor_input(resolution):
    with pytest.raises(ValueError, match="positive|divisible by patch_size"):
        load_and_preprocess_images(
            [torch.zeros(3, 32, 32, dtype=torch.uint8)],
            image_resolution=resolution,
            patch_size=16,
        )
