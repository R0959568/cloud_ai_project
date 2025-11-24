#!/bin/bash

echo "🚀 UK Housing Predictor - Quick Test"
echo "===================================="
echo ""

# Check if model exists
if [ ! -f "models/lightgbm_housing.pkl" ]; then
    echo "❌ Model file not found: models/lightgbm_housing.pkl"
    echo "   Copy your trained model to the models/ directory"
    exit 1
fi

# Check if data exists
if [ ! -f "data/cleaned/housing_FULL_clean.parquet" ]; then
    echo "⚠️  Data file not found: data/cleaned/housing_FULL_clean.parquet"
    echo "   Frontend dropdowns will use defaults"
fi

echo "✅ Files ready!"
echo ""
echo "Choose test mode:"
echo "1. Local (separate backend + frontend)"
echo "2. Docker (both services)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "📦 Installing backend dependencies..."
    cd backend
    pip install -q -r requirements.txt
    
    echo "🚀 Starting backend..."
    echo "   Backend will run at: http://localhost:8000"
    echo "   API docs at: http://localhost:8000/docs"
    echo ""
    echo "⚠️  Open a NEW terminal and run:"
    echo "   cd frontend"
    echo "   pip install -r requirements.txt"
    echo "   streamlit run app.py"
    echo ""
    echo "Press Ctrl+C to stop backend"
    echo ""
    uvicorn api:app --reload

elif [ "$choice" = "2" ]; then
    echo ""
    echo "🐳 Starting with Docker Compose..."
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:8501"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    docker-compose up --build

else
    echo "Invalid choice"
    exit 1
fi
