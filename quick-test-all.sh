#!/bin/bash

# Quick Test Script for Both Deployments
# ========================================

echo "🧪 Testing Housing and Electricity Deployments"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test Housing Backend
echo "📦 Testing Housing Backend (Port 8000)..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Housing backend is running${NC}"
else
    echo -e "${RED}❌ Housing backend is NOT running${NC}"
    echo "   Start with: cd housing-deployment && docker-compose up -d"
fi

# Test Housing Frontend
echo ""
echo "📦 Testing Housing Frontend (Port 8501)..."
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Housing frontend is running${NC}"
else
    echo -e "${RED}❌ Housing frontend is NOT running${NC}"
fi

echo ""
echo "⚡ Testing Electricity Backend (Port 8002)..."
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Electricity backend is running${NC}"
else
    echo -e "${RED}❌ Electricity backend is NOT running${NC}"
    echo "   Start with: cd electricity-deployment && docker-compose up -d"
fi

# Test Electricity Frontend
echo ""
echo "⚡ Testing Electricity Frontend (Port 8502)..."
if curl -s http://localhost:8502 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Electricity frontend is running${NC}"
else
    echo -e "${RED}❌ Electricity frontend is NOT running${NC}"
fi

echo ""
echo "=============================================="
echo "📊 Summary"
echo "=============================================="
echo ""
echo "If all services are running:"
echo "  🏠 Housing:      http://localhost:8501"
echo "  🏠 Housing API:  http://localhost:8000/docs"
echo "  ⚡ Electricity:  http://localhost:8502"
echo "  ⚡ Elec API:     http://localhost:8002/docs"
echo ""
echo "To start services:"
echo "  cd housing-deployment && docker-compose up -d"
echo "  cd electricity-deployment && docker-compose up -d"
echo ""
