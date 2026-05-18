"""
gradcam.py — Grad-CAM for PyTorch MobileNetV2 (binary classifier).

Hooks onto the last Conv2D layer, computes gradient-weighted feature maps,
and overlays a JET colourmap heatmap on the original BGR image.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ── Hook storage ─────────────────────────────────────────────────────────────
class _GradCAMHook:
    """Attaches forward + backward hooks to a target layer."""

    def __init__(self, layer):
        self.activations = None
        self.gradients   = None
        self._fwd = layer.register_forward_hook(self._save_activations)
        self._bwd = layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _, __, output):
        self.activations = output.detach()

    def _save_gradients(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def _find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last Conv2d layer in the model."""
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found in model.")
    return last


# ── Public API ────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(
    img_tensor: torch.Tensor,   # shape (1, 3, H, W), normalised, on device
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap.

    Returns
    -------
    heatmap : np.ndarray  shape (H, W), float32 in [0, 1]
    """
    model.eval()
    target_layer = _find_last_conv(model)
    hook = _GradCAMHook(target_layer)

    # Forward pass
    img_tensor = img_tensor.to(device).requires_grad_(False)
    logit = model(img_tensor).squeeze()          # scalar logit

    # Backward pass w.r.t. the prediction score
    model.zero_grad()
    logit.backward()

    hook.remove()

    # GAP over spatial dims -> channel weights
    grads       = hook.gradients   # (1, C, h, w)
    activations = hook.activations # (1, C, h, w)

    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam     = (weights * activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
    cam     = F.relu(cam)

    # Normalise
    cam = cam.squeeze().cpu().numpy()   # (h, w)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def overlay_gradcam(
    original_img_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap (JET colourmap) on a BGR image.

    Parameters
    ----------
    original_img_bgr : H x W x 3  uint8
    heatmap          : 2-D float in [0, 1]
    alpha            : blending weight for the heatmap

    Returns
    -------
    blended BGR image (uint8)
    """
    h, w = original_img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(original_img_bgr, 1 - alpha,
                           heatmap_colored, alpha, 0)