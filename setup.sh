#!/bin/bash

# Mesh Builder Setup Script
# This script sets up both backend and frontend using uv

set -e

echo "🎨 Setting up Mesh Builder..."
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv is required but not installed."
    echo "   Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

echo "✅ uv and Node.js found"
echo ""

# Backend setup
echo "📦 Setting up backend..."
cd backend
echo "Installing Python dependencies with uv..."
uv sync
cd ..

# Frontend setup
echo ""
echo "📦 Setting up frontend..."
cd frontend
echo "Installing Node dependencies..."
npm install --silent

cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo ""
echo "  Terminal 1 (Backend):"
echo "    cd backend && uv run python main.py"
echo ""
echo "  Terminal 2 (Frontend):"
echo "    cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser"
