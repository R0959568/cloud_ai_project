#!/bin/bash
# Railway startup script - starts all services

echo "🚀 Starting ML Prediction System on Railway..."

# Start housing backend in background
echo "Starting housing backend on port 8000..."
cd housing-deployment/backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &
HOUSING_PID=$!
cd ../..

# Start electricity backend in background  
echo "Starting electricity backend on port 8002..."
cd electricity-deployment/backend
python -m uvicorn api:app --host 0.0.0.0 --port 8002 &
ELECTRICITY_PID=$!
cd ../..

# Wait for backends to be ready
echo "Waiting for backends to initialize..."
sleep 5

# Check if backends are running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Housing backend ready"
else
    echo "⚠️ Housing backend not responding"
fi

if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ Electricity backend ready"
else
    echo "⚠️ Electricity backend not responding"
fi

# Start frontend (this will run in foreground)
echo "Starting combined frontend on port $PORT..."
export HOUSING_API_URL="http://localhost:8000"
export ELECTRICITY_API_URL="http://localhost:8002"

streamlit run combined_app.py \
    --server.port ${PORT:-8503} \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false

# If frontend exits, clean up background processes
kill $HOUSING_PID $ELECTRICITY_PID 2>/dev/null
