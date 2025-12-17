# Mesh Builder

AI-powered 3D mesh generation from images. Upload a single image or multiple views to generate downloadable 3D models.

## Features

- **Single Image to 3D**: Uses TripoSR (Stability AI) for high-quality reconstruction
- **Multi-View Support**: Combine multiple angles for better accuracy
- **Quality Settings**: Draft (fast), Standard (balanced), High (best quality)
- **Export Formats**: GLB, OBJ, PLY
- **Interactive Viewer**: Auto-rotate, wireframe mode, orbit controls

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **uv** (Python package manager)

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/melihucar/meshbuilder.git
cd meshbuilder
```

### 2. Install backend dependencies

```bash
cd backend
uv sync
cd ..
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Download TripoSR model weights (~1.5GB)

```bash
cd backend
./setup_triposr.sh
cd ..
```

This downloads:
- TripoSR repository
- Model weights from Hugging Face (stabilityai/TripoSR)

## Running the App

### Start both servers

```bash
./dev start
```

Open http://localhost:5173 in your browser.

### Other commands

```bash
./dev stop      # Stop all servers
./dev restart   # Restart all servers
./dev status    # Check server status
./dev logs      # View server logs
```

## Usage

1. **Select Mode**: Single Image or Multi-View
2. **Upload Image(s)**: Drag & drop or click to browse
3. **Choose Quality**: Draft (fast) / Standard / High (slow but detailed)
4. **Select Format**: GLB (recommended), OBJ, or PLY
5. **Generate**: Click "Generate 3D Mesh"
6. **View & Download**: Interact with the 3D viewer, then download

### Tips for Best Results

- Use images with **clean backgrounds** (or the AI will remove it)
- **Good lighting** improves depth estimation
- For multi-view: capture **front, back, left, right** angles
- **Center the object** in frame

## Tech Stack

### Backend
- FastAPI
- PyTorch + TripoSR
- rembg (background removal)
- PyMCubes (marching cubes mesh extraction)

### Frontend
- React 18 + TypeScript
- Vite
- TailwindCSS v4
- shadcn/ui
- Three.js + React Three Fiber

## Hardware Requirements

- **Minimum**: 8GB RAM, any modern CPU
- **Recommended**: 16GB RAM, Apple Silicon (MPS) or NVIDIA GPU (CUDA)

The app automatically detects and uses:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU (fallback, slower)

## Troubleshooting

### "Model not found" error
Run the setup script to download weights:
```bash
cd backend && ./setup_triposr.sh
```

### Slow generation on CPU
This is expected. For faster results:
- Use "Draft" quality
- Use a machine with GPU (CUDA or Apple Silicon MPS)

### Port already in use
```bash
./dev stop
./dev start
```

## License

MIT
