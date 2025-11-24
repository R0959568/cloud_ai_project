#!/bin/bash
# Start Combined ML Prediction System
# ====================================
# This script runs:
#   1. Housing backend on port 8000
#   2. Electricity backend on port 8002  
#   3. Combined frontend on port 8503

echo "🚀 Starting Combined ML Prediction System..."
echo "============================================"

# Stop any existing frontends (keep backends running)
echo "Stopping old frontends..."
pkill -f "streamlit run.*8501" 2>/dev/null
pkill -f "streamlit run.*8502" 2>/dev/null
sleep 1

# Check if backends are running
housing_running=$(curl -s http://localhost:8000/health 2>/dev/null | grep -q "healthy" && echo "yes" || echo "no")
electricity_running=$(curl -s http://localhost:8002/health 2>/dev/null | grep -q "healthy" && echo "yes" || echo "no")

# Start housing backend if not running
if [ "$housing_running" = "no" ]; then
    echo "▶️  Starting housing backend (port 8000)..."
    cd housing-deployment/backend
    python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 > /tmp/housing_backend.log 2>&1 &
    echo "   PID: $!"
    cd ../..
else
    echo "✅ Housing backend already running"
fi

# Start electricity backend if not running
if [ "$electricity_running" = "no" ]; then
    echo "▶️  Starting electricity backend (port 8002)..."
    cd electricity-deployment/backend
    python3 -m uvicorn api:app --host 0.0.0.0 --port 8002 > /tmp/electricity_backend.log 2>&1 &
    echo "   PID: $!"
    cd ../..
else
    echo "✅ Electricity backend already running"
fi

# Wait for backends
echo ""
echo "⏳ Waiting for backends to be ready..."
sleep 3

# Start combined frontend
echo "▶️  Starting combined frontend (port 8503)..."
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
python3 -m streamlit run combined_app.py \
    --server.port 8503 \
    --server.address 0.0.0.0 \
    > /tmp/combined_frontend.log 2>&1 &

echo "   PID: $!"

sleep 2

echo ""
echo "============================================"
echo "✅ COMBINED SYSTEM STARTED!"
echo "============================================"
echo ""
echo "🌐 Access Points:"
echo "   Combined Interface:  http://localhost:8503"
echo "   Housing API:         http://localhost:8000/docs"
echo "   Electricity API:     http://localhost:8002/docs"
echo ""
echo "📊 API Status:"
curl -s http://localhost:8000/health 2>/dev/null | grep -q "healthy" && echo "   ✅ Housing Backend:    Online" || echo "   ❌ Housing Backend:    Offline"
curl -s http://localhost:8002/health 2>/dev/null | grep -q "healthy" && echo "   ✅ Electricity Backend: Online" || echo "   ❌ Electricity Backend: Offline"
echo ""
echo "📝 Logs:"
echo "   Housing:     tail -f /tmp/housing_backend.log"
echo "   Electricity: tail -f /tmp/electricity_backend.log"
echo "   Frontend:    tail -f /tmp/combined_frontend.log"
echo ""
echo "🛑 To stop: ./stop-all.sh"
echo "============================================"
