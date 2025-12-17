"""
Compatibility layer: Replace torchmcubes with PyMCubes.

torchmcubes requires C++ compilation which fails on macOS ARM64.
This module provides a drop-in replacement using PyMCubes (pure Python).

Usage:
    import sys
    import torchmcubes_compat
    sys.modules['torchmcubes'] = torchmcubes_compat

    # Now any code that imports torchmcubes will get this module instead
"""
import mcubes
import torch
import numpy as np


def marching_cubes(volume: torch.Tensor, threshold: float):
    """
    Drop-in replacement for torchmcubes.marching_cubes.

    Extracts a triangle mesh from a 3D volume using the marching cubes algorithm.

    Args:
        volume: torch.Tensor of shape (D, H, W) - the 3D volume
        threshold: float - the isosurface threshold value

    Returns:
        vertices: torch.Tensor of shape (N, 3) - mesh vertex positions
        faces: torch.Tensor of shape (M, 3) - mesh face indices (triangles)
    """
    # Convert to numpy for PyMCubes
    # Always move to CPU first (handles CUDA, MPS, and CPU tensors)
    vol_np = volume.detach().cpu().numpy()

    # Run marching cubes
    vertices_np, faces_np = mcubes.marching_cubes(vol_np, threshold)

    # Convert back to torch tensors
    vertices = torch.from_numpy(vertices_np.astype(np.float32))
    faces = torch.from_numpy(faces_np.astype(np.int64))

    # Move to same device as input
    device = volume.device
    vertices = vertices.to(device)
    faces = faces.to(device)

    return vertices, faces


# Also export as the module-level function that torchmcubes uses
__all__ = ['marching_cubes']
