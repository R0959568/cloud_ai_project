#!/bin/bash

# Start Housing Backend and Frontend
# ==================================

echo "🏠 Starting Housing Prediction System..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if model exists
if [ ! -f "models/lightgbm_housing.pkl" ]; then
    echo "❌ Model file not found: models/lightgbm_housing.pkl"
    echo "   Please ensure the model file exists"
    exit 1
fi

# Start backend
echo "📡 Starting backend on port 8000..."
cd housing-deployment/backend
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 > /tmp/housing_backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
cd ../..

# Wait for backend to start
echo "   Waiting for backend to start..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running!${NC}"
else
    echo "⚠️  Backend may not be fully started yet"
    echo "   Check logs: tail -f /tmp/housing_backend.log"
fi

# Start frontend
echo ""
echo "🌐 Starting frontend on port 8501..."
cd housing-deployment/frontend
python3 -m streamlit run app.py --server.port 8501 > /tmp/housing_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
cd ../..

echo ""
echo -e "${GREEN}✅ Housing system started!${NC}"
echo ""
echo "Access points:"
echo "  Frontend: http://localhost:8501"
echo "  Backend:  http://localhost:8000/docs"
echo ""
echo "Process IDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/housing_backend.log"
echo "  Frontend: tail -f /tmp/housing_frontend.log"
echo ""
echo "To stop:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
