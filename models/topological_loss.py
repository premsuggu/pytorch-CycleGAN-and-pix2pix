import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gudhi as gd

def normalized_to_grayscale(batch: torch.Tensor) -> torch.Tensor:
    """Converts CycleGAN normalized tensor [-1, 1] to Grayscale [0, 1]."""
    batch_01 = (batch.clamp(-1.0, 1.0) + 1.0) / 2.0
    if batch_01.shape[1] == 1:
        return batch_01
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch_01[:, :3] * weights).sum(dim=1, keepdim=True)

def _filter_topology_points(diagram: np.ndarray, threshold: float = 0.01, max_points: int = 50) -> np.ndarray:
    """Cleans persistence diagrams by removing short-lived noise and limiting point count."""
    if diagram.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    mask = np.isfinite(diagram[:, 1]) & ((diagram[:, 1] - diagram[:, 0]) > threshold)
    filtered = diagram[mask]
    if filtered.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    persistence = filtered[:, 1] - filtered[:, 0]
    keep = np.argsort(persistence)[-max_points:]
    return filtered[keep].astype(np.float64, copy=False)

def _persistence_diagrams(image_2d: np.ndarray, threshold: float = 0.01, max_points: int = 50):
    """Computes persistence diagrams for H0 and H1 using Gudhi Cubical Complex."""
    filt_arr = image_2d.astype(np.float64)
    complex_ = gd.CubicalComplex(dimensions=filt_arr.shape, top_dimensional_cells=filt_arr.flatten())
    complex_.persistence()
    dim0 = _filter_topology_points(complex_.persistence_intervals_in_dimension(0), threshold, max_points)
    dim1 = _filter_topology_points(complex_.persistence_intervals_in_dimension(1), threshold, max_points)
    return dim0, dim1

def _safe_wasserstein(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Calculates 1-Wasserstein distance between two persistence diagrams."""
    if dgm_a.size == 0 and dgm_b.size == 0:
        return 0.0
    dgm_a = dgm_a.reshape(-1, 2)
    dgm_b = dgm_b.reshape(-1, 2)
    return float(gd.wasserstein.wasserstein_distance(dgm_a, dgm_b, order=1.0, keep_essential_parts=False))

def _sobel_edges(gray_batch: torch.Tensor) -> torch.Tensor:
    """Differentiable edge detection for structural surrogate loss."""
    kernel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=gray_batch.device, dtype=gray_batch.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=gray_batch.device, dtype=gray_batch.dtype).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray_batch, kernel_x, padding=1)
    grad_y = F.conv2d(gray_batch, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

class WassersteinTopologicalLoss(nn.Module):
    """
    Gudhi-based Topological Loss with Straight-Through Estimator.
    Computes H0 (components) and H1 (loops) Wasserstein distance.
    """
    def __init__(self, topo_resize=128, weight_dim0=1.0, weight_dim1=1.0, threshold=0.01, max_points=50, surrogate_weight=0.5):
        super().__init__()
        self.topo_resize = topo_resize
        self.weight_dim0 = weight_dim0
        self.weight_dim1 = weight_dim1
        self.threshold = threshold
        self.max_points = max_points
        self.surrogate_weight = surrogate_weight

    def topology_value(self, fake: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        fake_gray = normalized_to_grayscale(fake)
        ref_gray = normalized_to_grayscale(reference)
        if self.topo_resize is not None and fake_gray.shape[-1] != self.topo_resize:
            fake_small = F.interpolate(fake_gray, size=(self.topo_resize, self.topo_resize), mode="bilinear", align_corners=False)
            ref_small = F.interpolate(ref_gray, size=(self.topo_resize, self.topo_resize), mode="bilinear", align_corners=False)
        else:
            fake_small = fake_gray
            ref_small = ref_gray

        topo_values = []
        for fake_img, ref_img in zip(fake_small, ref_small):
            fake_np = fake_img.squeeze(0).detach().cpu().numpy().astype(np.float64)
            ref_np = ref_img.squeeze(0).detach().cpu().numpy().astype(np.float64)
            fake_d0, fake_d1 = _persistence_diagrams(fake_np, self.threshold, self.max_points)
            ref_d0, ref_d1 = _persistence_diagrams(ref_np, self.threshold, self.max_points)
            score = self.weight_dim0 * _safe_wasserstein(fake_d0, ref_d0) + self.weight_dim1 * _safe_wasserstein(fake_d1, ref_d1)
            topo_values.append(score)

        return torch.tensor(float(np.mean(topo_values)), device=fake.device, dtype=fake.dtype)

    def surrogate(self, fake: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        fake_gray = normalized_to_grayscale(fake)
        ref_gray = normalized_to_grayscale(reference)
        if self.topo_resize is not None and fake_gray.shape[-1] != self.topo_resize:
            fake_small = F.interpolate(fake_gray, size=(self.topo_resize, self.topo_resize), mode="bilinear", align_corners=False)
            ref_small = F.interpolate(ref_gray, size=(self.topo_resize, self.topo_resize), mode="bilinear", align_corners=False)
        else:
            fake_small = fake_gray
            ref_small = ref_gray

        edge_term = F.l1_loss(_sobel_edges(fake_small), _sobel_edges(ref_small))
        intensity_term = F.l1_loss(fake_small, ref_small)
        return edge_term + self.surrogate_weight * intensity_term

    def forward(self, fake: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        topo_val = self.topology_value(fake, reference)
        surr_val = self.surrogate(fake, reference)
        # STE Trick: Value from TDA, Gradients from Surrogate
        return surr_val + (topo_val - surr_val.detach())
