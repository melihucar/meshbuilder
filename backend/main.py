"""
Mesh Builder API - Generate 3D meshes from images using AI
"""
import io
import os
import uuid
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import MeshGenerator, MultiViewGenerator

app = FastAPI(
    title="Mesh Builder API",
    description="Generate 3D meshes from images using TripoSR",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory for generated meshes
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for serving generated meshes
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Initialize models (lazy loading)
mesh_generator: MeshGenerator | None = None
multiview_generator: MultiViewGenerator | None = None


def get_generator() -> MeshGenerator:
    """Get or initialize the single-image mesh generator."""
    global mesh_generator
    if mesh_generator is None:
        mesh_generator = MeshGenerator()
    return mesh_generator


def get_multiview_generator() -> MultiViewGenerator:
    """Get or initialize the multi-view mesh generator."""
    global multiview_generator
    if multiview_generator is None:
        multiview_generator = MultiViewGenerator()
    return multiview_generator


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Mesh Builder API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": mesh_generator is not None,
    }


@app.post("/generate")
async def generate_mesh(
    image: UploadFile = File(..., description="Image file to convert to 3D mesh"),
    format: str = "glb",
    quality: str = "standard"
):
    """
    Generate a 3D mesh from an uploaded image.

    Args:
        image: The image file (PNG, JPG, WEBP)
        format: Output format (glb, obj, ply)
        quality: Quality level - draft (fast), standard (balanced), high (best quality)

    Returns:
        JSON with mesh file URL and metadata
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/webp"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    # Validate format
    allowed_formats = ["glb", "obj", "ply"]
    if format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Allowed: {allowed_formats}"
        )

    # Validate quality
    allowed_qualities = ["draft", "standard", "high"]
    if quality not in allowed_qualities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quality. Allowed: {allowed_qualities}"
        )

    try:
        # Read and process image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Generate unique ID for this mesh
        mesh_id = str(uuid.uuid4())[:8]
        output_filename = f"mesh_{mesh_id}.{format}"
        output_path = OUTPUT_DIR / output_filename

        # Generate mesh
        generator = get_generator()
        generator.generate(pil_image, str(output_path), format=format, quality=quality)

        return JSONResponse({
            "success": True,
            "mesh_id": mesh_id,
            "mesh_url": f"/outputs/{output_filename}",
            "format": format,
            "quality": quality,
            "message": "Mesh generated successfully"
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate mesh: {str(e)}"
        )


@app.post("/generate-multiview")
async def generate_multiview_mesh(
    images: List[UploadFile] = File(..., description="Multiple images from different angles"),
    format: str = "glb"
):
    """
    Generate a 3D mesh from multiple images (multi-view reconstruction).

    For best results, provide 4-6 images of the object from different angles:
    - Front, back, left, right views
    - Optional: top and 45-degree angles

    Args:
        images: List of image files (PNG, JPG, WEBP)
        format: Output format (glb, obj, ply)

    Returns:
        JSON with mesh file URL and metadata
    """
    # Validate number of images
    if len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 images required for multi-view reconstruction"
        )

    if len(images) > 12:
        raise HTTPException(
            status_code=400,
            detail="Maximum 12 images allowed"
        )

    # Validate file types
    allowed_types = ["image/png", "image/jpeg", "image/webp"]
    for img in images:
        if img.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type for {img.filename}. Allowed: {allowed_types}"
            )

    # Validate format
    allowed_formats = ["glb", "obj", "ply"]
    if format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Allowed: {allowed_formats}"
        )

    try:
        # Read and process all images
        pil_images = []
        for img in images:
            contents = await img.read()
            pil_image = Image.open(io.BytesIO(contents))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_images.append(pil_image)

        # Generate unique ID for this mesh
        mesh_id = str(uuid.uuid4())[:8]
        output_filename = f"mesh_{mesh_id}.{format}"
        output_path = OUTPUT_DIR / output_filename

        # Generate mesh from multiple views
        generator = get_multiview_generator()
        generator.generate(pil_images, str(output_path), format=format)

        return JSONResponse({
            "success": True,
            "mesh_id": mesh_id,
            "mesh_url": f"/outputs/{output_filename}",
            "format": format,
            "num_images": len(images),
            "message": "Multi-view mesh generated successfully"
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate mesh: {str(e)}"
        )


@app.get("/download/{mesh_id}")
async def download_mesh(mesh_id: str, format: str = "glb"):
    """Download a generated mesh file."""
    filename = f"mesh_{mesh_id}.{format}"
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="model/gltf-binary" if format == "glb" else "application/octet-stream"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
