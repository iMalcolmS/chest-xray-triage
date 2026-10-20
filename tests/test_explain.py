import numpy as np
import torch
import torchxrayvision as xrv

from cxr.explain import Logits, fraction_in_mask, gradcam


def test_fraction_in_mask():
    heat = np.zeros((2, 4, 4))
    heat[0, :2, :] = 1.0          # all weight in the top half
    heat[1] = 1.0                 # uniform
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[:, :2, :] = True         # mask = top half
    np.testing.assert_allclose(fraction_in_mask(heat, mask), [1.0, 0.5])


def test_fraction_in_mask_empty_heatmap_is_zero():
    assert fraction_in_mask(np.zeros((1, 3, 3)), np.ones((1, 3, 3), bool))[0] == 0


def _untrained_densenet():
    return xrv.models.DenseNet(weights=None, num_classes=3).eval()


def test_logits_wrapper_matches_classifier():
    model = _untrained_densenet()
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        expected = model.classifier(model.features2(x))
        np.testing.assert_allclose(Logits(model)(x).numpy(), expected.numpy(), rtol=1e-5)


def test_gradcam_shape_and_range():
    heat = gradcam(_untrained_densenet(), torch.randn(2, 1, 224, 224), output_index=1)
    assert heat.shape == (2, 224, 224)
    assert heat.min() >= 0 and heat.max() <= 1
