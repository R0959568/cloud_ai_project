#!/bin/bash

# Stop All Services
# ================

echo "🛑 Stopping all services..."

# Kill housing services
pkill -f "uvicorn api:app --host 0.0.0.0 --port 8000"
pkill -f "streamlit run app.py --server.port 8501"

# Kill electricity services  
pkill -f "uvicorn api:app --host 0.0.0.0 --port 8002"
pkill -f "streamlit run app.py --server.port 8502"

echo "✅ All services stopped"

# Clean up log files
rm -f /tmp/housing_backend.log /tmp/housing_frontend.log
rm -f /tmp/electricity_backend.log /tmp/electricity_frontend.log

echo "✅ Log files cleaned up"
