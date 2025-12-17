"""
3D Mesh Generation using AI models.

Supports:
- TripoSR: Single-image 3D reconstruction (Stability AI) - Best quality
- Depth-based: Fallback using monocular depth estimation
"""
import os
import sys
import logging
from pathlib import Path
from typing import Literal, List, Optional

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add TripoSR to path if cloned locally
TRIPOSR_PATH = Path(__file__).parent / "triposr"
if TRIPOSR_PATH.exists():
    sys.path.insert(0, str(TRIPOSR_PATH))

# Inject torchmcubes compatibility shim BEFORE any TSR imports
# This replaces torchmcubes (which requires C++ compilation) with PyMCubes
try:
    from . import torchmcubes_compat
    sys.modules['torchmcubes'] = torchmcubes_compat
    logger.info("Injected torchmcubes compatibility shim (using PyMCubes)")
except ImportError:
    # Running as standalone script
    import torchmcubes_compat
    sys.modules['torchmcubes'] = torchmcubes_compat
    logger.info("Injected torchmcubes compatibility shim (using PyMCubes)")


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image using rembg."""
    try:
        from rembg import remove, new_session

        logger.info("Removing background...")
        # Use u2net model for best quality
        session = new_session("u2net")
        result = remove(image, session=session)

        # Ensure RGBA
        if result.mode != "RGBA":
            result = result.convert("RGBA")

        logger.info("Background removed successfully")
        return result

    except ImportError:
        logger.warning("rembg not installed, skipping background removal")
        return image.convert("RGBA") if image.mode != "RGBA" else image
    except Exception as e:
        logger.warning(f"Background removal failed: {e}")
        return image.convert("RGBA") if image.mode != "RGBA" else image


def preprocess_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Preprocess image for 3D generation."""
    # Remove background
    image = remove_background(image)

    # Resize while maintaining aspect ratio
    image.thumbnail((size, size), Image.Resampling.LANCZOS)

    # Create square canvas and center the image
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    paste_x = (size - image.width) // 2
    paste_y = (size - image.height) // 2
    canvas.paste(image, (paste_x, paste_y), image if image.mode == "RGBA" else None)

    return canvas


class MeshGenerator:
    """
    Generate 3D meshes from single images.

    Tries to use TripoSR for best quality, falls back to depth-based generation.
    """

    def __init__(self, device: str | None = None):
        self.device = self._get_device(device)
        self.model = None
        self.model_type = None
        self._load_model()

    def _get_device(self, device: str | None) -> str:
        """Detect the best available device."""
        if device:
            return device

        if torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS")
            return "mps"
        else:
            logger.info("Using CPU (this will be slow)")
            return "cpu"

    def _load_model(self):
        """Load the best available model."""
        # Try TripoSR first
        if self._try_load_triposr():
            return

        # Fall back to depth estimation
        self._setup_depth_fallback()

    def _try_load_triposr(self) -> bool:
        """Try to load TripoSR model."""
        try:
            # Check if TripoSR is available
            from tsr.system import TSR

            logger.info("Loading TripoSR model...")

            # Check for local pretrained weights first
            local_pretrained = TRIPOSR_PATH / "pretrained"
            if local_pretrained.exists() and (local_pretrained / "model.ckpt").exists():
                logger.info(f"Using local weights from {local_pretrained}")
                self.model = TSR.from_pretrained(
                    str(local_pretrained),
                    config_name="config.yaml",
                    weight_name="model.ckpt"
                )
            else:
                # Download from Hugging Face
                logger.info("Downloading weights from Hugging Face...")
                self.model = TSR.from_pretrained(
                    "stabilityai/TripoSR",
                    config_name="config.yaml",
                    weight_name="model.ckpt"
                )

            self.model.renderer.set_chunk_size(8192)
            self.model.to(self.device)
            self.model_type = "triposr"

            logger.info("TripoSR model loaded successfully!")
            return True

        except ImportError as e:
            logger.info(f"TripoSR not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to load TripoSR: {e}")
            return False

    def _setup_depth_fallback(self):
        """Set up depth estimation fallback."""
        try:
            from transformers import pipeline

            logger.info("Loading depth estimation model (fallback)...")

            # Use DPT for depth estimation
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device == "cuda" else -1
            )
            self.model_type = "depth"
            logger.info("Depth estimation model loaded")

        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.model_type = "placeholder"

    def generate(
        self,
        image: Image.Image,
        output_path: str,
        format: Literal["glb", "obj", "ply"] = "glb",
        remove_bg: bool = True,
        quality: Literal["draft", "standard", "high"] = "standard"
    ) -> str:
        """
        Generate a 3D mesh from an image.

        Args:
            image: PIL Image to convert
            output_path: Path to save the mesh
            format: Output format (glb, obj, ply)
            remove_bg: Whether to remove background first
            quality: Quality level - draft (256), standard (512), high (768)

        Returns:
            Path to the generated mesh file
        """
        # Map quality to resolution
        resolution_map = {"draft": 256, "standard": 512, "high": 768}
        resolution = resolution_map.get(quality, 512)

        logger.info(f"Generating mesh using {self.model_type} model (quality={quality}, resolution={resolution})...")

        # Preprocess image
        if remove_bg:
            processed = preprocess_image(image)
        else:
            processed = image.resize((512, 512), Image.Resampling.LANCZOS)

        if self.model_type == "triposr":
            return self._generate_triposr(processed, output_path, format, resolution)
        elif self.model_type == "depth":
            return self._generate_from_depth(processed, output_path, format)
        else:
            return self._generate_placeholder(processed, output_path, format)

    def _generate_triposr(
        self,
        image: Image.Image,
        output_path: str,
        format: str,
        resolution: int = 512
    ) -> str:
        """Generate mesh using TripoSR - produces true 3D meshes."""
        logger.info(f"Running TripoSR inference (resolution={resolution})...")

        # Convert RGBA to RGB with white background for TripoSR
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        with torch.no_grad():
            scene_codes = self.model([image], device=self.device)

        logger.info("Extracting mesh...")
        # has_vertex_color=True to get colored mesh
        meshes = self.model.extract_mesh(scene_codes, True, resolution=resolution)
        mesh = meshes[0]

        # Export mesh
        mesh.export(output_path)
        logger.info(f"Mesh saved to {output_path}")

        return output_path

    def _generate_from_depth(
        self,
        image: Image.Image,
        output_path: str,
        format: str
    ) -> str:
        """Generate mesh from depth estimation - produces 2.5D relief."""
        import trimesh

        # Convert to RGB for depth estimation
        if image.mode == "RGBA":
            # Keep alpha for masking
            alpha = np.array(image.split()[3]) / 255.0
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            rgb_image = background
        else:
            alpha = None
            rgb_image = image

        width, height = rgb_image.size

        # Estimate depth
        logger.info("Estimating depth...")
        result = self.depth_estimator(rgb_image)
        depth = np.array(result["depth"])

        # Normalize depth to 0-1 range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Create mesh from depth map
        logger.info("Creating mesh from depth...")
        mesh = self._depth_to_mesh(depth, np.array(rgb_image), width, height, alpha)

        # Export
        mesh.export(output_path, file_type=format)
        logger.info(f"Mesh saved to {output_path}")

        return output_path

    def _depth_to_mesh(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        width: int,
        height: int,
        alpha: Optional[np.ndarray] = None,
        depth_scale: float = 0.4
    ):
        """Convert depth map to mesh with proper handling."""
        import trimesh

        # Create coordinate grids
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Create vertices
        vertices = np.zeros((height * width, 3))
        vertices[:, 0] = xx.flatten()
        vertices[:, 1] = -yy.flatten()
        vertices[:, 2] = depth.flatten() * depth_scale

        # Create faces (triangles)
        faces = []
        valid_mask = np.ones(height * width, dtype=bool)

        # If we have alpha, only include vertices with alpha > 0.5
        if alpha is not None:
            valid_mask = alpha.flatten() > 0.5

        for i in range(height - 1):
            for j in range(width - 1):
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)

                # Only create faces where all vertices are valid
                if alpha is not None:
                    if not (valid_mask[v0] and valid_mask[v1] and valid_mask[v2]):
                        continue
                    if not (valid_mask[v1] and valid_mask[v2] and valid_mask[v3]):
                        continue

                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])

        faces = np.array(faces) if faces else np.zeros((0, 3), dtype=int)

        # Get colors from image
        colors = image.reshape(-1, 3)
        alpha_channel = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors = np.hstack([colors, alpha_channel])

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            process=True  # Auto-clean mesh on creation
        )

        return mesh

    def _generate_placeholder(
        self,
        image: Image.Image,
        output_path: str,
        format: str
    ) -> str:
        """Generate a placeholder mesh when no ML models available."""
        import trimesh

        logger.warning("No ML model available - generating placeholder")

        mesh = trimesh.creation.box(extents=[1, 1, 1])

        avg_color = np.array(image.resize((1, 1)).convert("RGB").getpixel((0, 0)))
        avg_color = np.append(avg_color, 255)
        mesh.visual.vertex_colors = avg_color

        mesh.export(output_path, file_type=format)
        return output_path


class MultiViewGenerator:
    """
    Generate 3D meshes from multiple images using multi-view reconstruction.
    """

    def __init__(self, device: str | None = None):
        self.device = self._get_device(device)
        self.model_type = None
        self._load_model()

    def _get_device(self, device: str | None) -> str:
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load multi-view reconstruction model."""
        try:
            from transformers import pipeline

            logger.info("Loading depth model for multi-view fusion...")
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device == "cuda" else -1
            )
            self.model_type = "multiview_depth"
            logger.info("Multi-view depth fusion ready")

        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.model_type = "placeholder"

    def generate(
        self,
        images: List[Image.Image],
        output_path: str,
        format: Literal["glb", "obj", "ply"] = "glb"
    ) -> str:
        """Generate mesh from multiple view images."""
        logger.info(f"Generating mesh from {len(images)} views...")

        # Preprocess all images
        processed_images = []
        for img in images:
            processed = preprocess_image(img, size=256)
            processed_images.append(processed)

        if self.model_type == "multiview_depth":
            return self._generate_multiview_depth(processed_images, output_path, format)
        else:
            return self._generate_placeholder(processed_images, output_path, format)

    def _generate_multiview_depth(
        self,
        images: List[Image.Image],
        output_path: str,
        format: str
    ) -> str:
        """Generate mesh by fusing depth from multiple views."""
        import trimesh

        all_points = []
        all_colors = []

        for i, image in enumerate(images):
            logger.info(f"Processing view {i + 1}/{len(images)}...")

            # Convert to RGB
            if image.mode == "RGBA":
                alpha = np.array(image.split()[3]) / 255.0
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                rgb_image = background
            else:
                alpha = np.ones((image.height, image.width))
                rgb_image = image

            width, height = rgb_image.size

            # Estimate depth
            result = self.depth_estimator(rgb_image)
            depth = np.array(result["depth"])
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # Create points
            points, colors = self._view_to_points(
                depth, np.array(rgb_image), alpha, width, height,
                view_index=i, num_views=len(images)
            )

            all_points.append(points)
            all_colors.append(colors)

        # Combine all points
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)

        # Create mesh from point cloud
        logger.info("Creating mesh from combined views...")

        try:
            # Try to create a proper mesh using convex hull
            cloud = trimesh.PointCloud(combined_points, colors=combined_colors)
            mesh = cloud.convex_hull

            # Interpolate colors
            from scipy.spatial import cKDTree
            tree = cKDTree(combined_points)
            _, indices = tree.query(mesh.vertices, k=1)
            mesh.visual.vertex_colors = combined_colors[indices]

        except Exception as e:
            logger.warning(f"Convex hull failed: {e}, using point cloud")
            # Export as point cloud in PLY format
            mesh = trimesh.PointCloud(combined_points, colors=combined_colors)

        mesh.export(output_path, file_type=format)
        logger.info(f"Multi-view mesh saved to {output_path}")

        return output_path

    def _view_to_points(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        alpha: np.ndarray,
        width: int,
        height: int,
        view_index: int,
        num_views: int
    ):
        """Convert a single view to 3D points."""
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        points = np.zeros((height * width, 3))
        points[:, 0] = xx.flatten()
        points[:, 1] = -yy.flatten()
        points[:, 2] = depth.flatten() * 0.5

        # Rotate based on view index
        if num_views > 1:
            angle = (2 * np.pi * view_index) / num_views
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            points = points @ rotation.T

        # Get colors
        colors = image.reshape(-1, 3)
        alpha_flat = (alpha.flatten() * 255).astype(np.uint8)
        colors = np.hstack([colors, alpha_flat.reshape(-1, 1)])

        # Filter by alpha
        mask = alpha.flatten() > 0.5
        return points[mask], colors[mask]

    def _generate_placeholder(
        self,
        images: List[Image.Image],
        output_path: str,
        format: str
    ) -> str:
        """Placeholder for when no model is available."""
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh.export(output_path, file_type=format)
        return output_path


if __name__ == "__main__":
    # Test
    print("Testing MeshGenerator...")
    gen = MeshGenerator()
    print(f"Model type: {gen.model_type}")

    test_img = Image.new("RGB", (256, 256), color="blue")
    gen.generate(test_img, "test_output.glb")
    print("Done!")
