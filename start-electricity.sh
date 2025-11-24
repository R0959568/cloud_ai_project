#!/bin/bash

# Start Electricity Backend and Frontend
# ======================================

echo "⚡ Starting Electricity Demand Prediction System..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if model exists
if [ ! -f "dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl" ]; then
    echo "❌ Model file not found: dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl"
    echo "   Please ensure the model file exists"
    exit 1
fi

# Start backend
echo "📡 Starting backend on port 8002..."
cd electricity-deployment/backend
python3 -m uvicorn api:app --host 0.0.0.0 --port 8002 > /tmp/electricity_backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
cd ../..

# Wait for backend to start
echo "   Waiting for backend to start..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running!${NC}"
else
    echo "⚠️  Backend may not be fully started yet"
    echo "   Check logs: tail -f /tmp/electricity_backend.log"
fi

# Start frontend
echo ""
echo "🌐 Starting frontend on port 8502..."
cd electricity-deployment/frontend

# Temporarily set API_URL for local testing
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
python3 -m streamlit run app.py --server.port 8502 > /tmp/electricity_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
cd ../..

echo ""
echo -e "${GREEN}✅ Electricity system started!${NC}"
echo ""
echo "Access points:"
echo "  Frontend: http://localhost:8502"
echo "  Backend:  http://localhost:8002/docs"
echo ""
echo "Process IDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/electricity_backend.log"
echo "  Frontend: tail -f /tmp/electricity_frontend.log"
echo ""
echo "To stop:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
