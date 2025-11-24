# 🏠⚡ UK Housing & Electricity Prediction - Dual Deployment System

**Complete ML Deployment Project with FastAPI, Streamlit, and Docker**

Team: Error400
- Hamid Iqbal
- Ibrahim Afkir

---

## 📊 Project Overview

This project implements **TWO complete ML prediction systems**, each with professional backend/frontend separation and Docker deployment:

### **Dataset 1: UK Housing Price Prediction** 🏠
- **Model:** LightGBM Regressor (R² ~67%)
- **Data:** 5.9M UK housing transactions (1995-2017)
- **Features:** Location (county, district, town), property type, tenure, date
- **Deployment:** FastAPI backend + Streamlit frontend

### **Dataset 2: UK Electricity Demand Prediction** ⚡
- **Model:** Gradient Boosting Regressor (R² 0.70)
- **Data:** UK electricity demand (2001-2025)
- **Features:** 39 enhanced features (lag, rolling stats, temporal, holidays)
- **Deployment:** FastAPI backend + Streamlit frontend

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT STRUCTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📦 Dataset 1: Housing (Port 8000/8501)                     │
│  ┌────────────┐      ┌──────────┐      ┌────────────┐     │
│  │ Streamlit  │ ───▶ │ FastAPI  │ ───▶ │  LightGBM  │     │
│  │  Frontend  │ HTTP │  Backend │      │   Model    │     │
│  │  :8501     │      │  :8000   │      │   .pkl     │     │
│  └────────────┘      └──────────┘      └────────────┘     │
│                                                              │
│  📦 Dataset 2: Electricity (Port 8002/8502)                 │
│  ┌────────────┐      ┌──────────┐      ┌────────────┐     │
│  │ Streamlit  │ ───▶ │ FastAPI  │ ───▶ │  Gradient  │     │
│  │  Frontend  │ HTTP │  Backend │      │  Boosting  │     │
│  │  :8502     │      │  :8002   │      │   Model    │     │
│  └────────────┘      └──────────┘      └────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
cloud_ai_project-main/
├── housing-deployment/              # Dataset 1 Deployment
│   ├── backend/
│   │   ├── api.py                  # FastAPI backend
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── app.py                  # Streamlit frontend
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── docker-compose.yml
│   └── DEPLOYMENT_GUIDE.md
│
├── electricity-deployment/          # Dataset 2 Deployment
│   ├── backend/
│   │   ├── api.py                  # FastAPI backend (39 features)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── app.py                  # Streamlit frontend
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── docker-compose.yml
│
├── models/
│   └── lightgbm_housing.pkl        # Housing model
│
├── dataset_2_electricity_app/
│   └── data/final/models/
│       └── gradient_boosting_enhanced.pkl  # Electricity model
│
└── README.md                        # This file
```

---

## 🚀 Quick Start - Local Testing

### **Prerequisites**
- Python 3.10+
- Docker & Docker Compose (for containerized deployment)

### **Option 1: Test WITHOUT Docker (Fastest)**

#### **Housing Deployment:**
```bash
# Terminal 1 - Backend
cd housing-deployment/backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000

# Terminal 2 - Frontend
cd housing-deployment/frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```
Access: http://localhost:8501

#### **Electricity Deployment:**
```bash
# Terminal 1 - Backend
cd electricity-deployment/backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8002

# Terminal 2 - Frontend
cd electricity-deployment/frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```
Access: http://localhost:8502

---

### **Option 2: Test WITH Docker (Production-like)**

#### **Housing Deployment:**
```bash
cd housing-deployment
docker-compose up --build
```
- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:8501

#### **Electricity Deployment:**
```bash
cd electricity-deployment
docker-compose up --build
```
- Backend API: http://localhost:8002/docs
- Frontend: http://localhost:8502

---

### **Option 3: Run BOTH Systems Simultaneously**

```bash
# Terminal 1 - Housing
cd housing-deployment
docker-compose up

# Terminal 2 - Electricity
cd electricity-deployment
docker-compose up
```

Access both systems:
- **Housing:** http://localhost:8501
- **Electricity:** http://localhost:8502

---

## 🔧 Technical Details

### **Dataset 1: Housing Price Prediction**

**API Endpoints:**
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Predict house price
- `GET /model-info` - Model information

**Request Example:**
```json
{
  "property_type_label": "Detached",
  "is_new_build": false,
  "tenure_label": "Freehold",
  "county": "GREATER LONDON",
  "district": "CITY OF WESTMINSTER",
  "town_city": "LONDON",
  "year": 2017,
  "month": 6,
  "quarter": 2
}
```

**Response Example:**
```json
{
  "predicted_price": 450000.0,
  "lower_bound": 405000.0,
  "upper_bound": 495000.0,
  "message": "Prediction successful"
}
```

---

### **Dataset 2: Electricity Demand Prediction**

**API Endpoints:**
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Predict electricity demand
- `GET /model-info` - Model information (39 features)

**Request Example:**
```json
{
  "prediction_datetime": "2024-06-15T14:30:00"
}
```

**Response Example:**
```json
{
  "predicted_demand_mw": 35420.5,
  "prediction_datetime": "2024-06-15T14:30:00",
  "features_used": {
    "year": 2024,
    "month": 6,
    "hour": 14,
    "is_weekend": false,
    "season": 2,
    "demand_lag_1d": 34800.0,
    "rolling_mean_24h": 35100.0
  },
  "message": "Prediction successful"
}
```

**Feature Categories:**
- **Temporal:** year, month, day, hour, day_of_week, quarter, week_of_year (7)
- **Binary Indicators:** is_weekend, is_business_hours, is_night, peak indicators (5)
- **Cyclical Encoding:** hour_sin/cos, month_sin/cos, day_of_week_sin/cos (6)
- **Lag Features:** demand_lag_1, demand_lag_1d, demand_lag_3h, demand_lag_7d (4)
- **Rolling Statistics:** 24h/7d means, std, diff from average (4)
- **Holidays:** is_holiday, day_before/after_holiday (3)
- **Interactions:** weekend_hour, holiday_hour, month_hour (3)
- **Total: 39 features**

---

## 🌐 Cloud Deployment

### **Oracle Cloud Free Tier (Recommended)**

Both systems can be deployed on Oracle Cloud free tier VMs. Each system uses ~1GB RAM.

#### **Deployment Strategy:**

**Option A: Single VM (Both Systems)**
- Use 1 VM with 1GB RAM (free tier)
- Run both Docker Compose setups
- Ports: Housing (8000/8501), Electricity (8002/8502)

**Option B: Separate VMs (Recommended)**
- VM 1: Housing prediction system
- VM 2: Electricity prediction system
- Each VM runs one docker-compose setup

#### **Quick Deploy Commands:**

```bash
# SSH into Oracle VM
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

# Clone repository
git clone YOUR_REPO_URL
cd cloud_ai_project-main

# Deploy Housing
cd housing-deployment
docker-compose up -d

# Deploy Electricity (same VM or different)
cd ../electricity-deployment
docker-compose up -d
```

#### **Firewall Configuration:**
Open these ports in Oracle Cloud Security Lists:
- **Housing:** 8000 (API), 8501 (Frontend)
- **Electricity:** 8002 (API), 8502 (Frontend)

---

## 📊 Model Performance

### **Housing Price Prediction**
- **Model:** LightGBM Regressor
- **R² Score:** 0.67
- **Training Data:** 5.9M transactions
- **Features:** 9 (location, property characteristics, date)

### **Electricity Demand Prediction**
- **Model:** Gradient Boosting Regressor
- **R² Score:** 0.70
- **MAE:** 2,353 MW
- **RMSE:** 3,107 MW
- **Training Data:** 2001-2025 (25 years)
- **Features:** 39 (enhanced with lag, rolling stats, holidays)

---

## 🧪 Testing

### **Test Backend API (Housing):**
```bash
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "property_type_label": "Detached",
    "is_new_build": false,
    "tenure_label": "Freehold",
    "county": "GREATER LONDON",
    "district": "CITY OF WESTMINSTER",
    "town_city": "LONDON",
    "year": 2017,
    "month": 6,
    "quarter": 2
  }'
```

### **Test Backend API (Electricity):**
```bash
curl http://localhost:8002/health

curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"prediction_datetime": "2024-06-15T14:30:00"}'
```

---

## 🔍 Troubleshooting

### **Backend won't start:**
```bash
# Check logs
docker logs housing-backend
docker logs electricity-backend

# Verify model files exist
ls -lh models/lightgbm_housing.pkl
ls -lh dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl
```

### **Frontend can't reach backend:**
```bash
# Check if backend is running
curl http://localhost:8000/health  # Housing
curl http://localhost:8002/health  # Electricity

# Update API_URL in frontend/app.py if needed
# Docker: API_URL = "http://backend:8000"
# Local: API_URL = "http://localhost:8000"
```

### **Port conflicts:**
```bash
# Stop all containers
docker-compose down

# Change ports in docker-compose.yml
# Housing: 8000→8000, 8501→8501
# Electricity: 8002→8002, 8502→8502
```

---

## 📝 Development Notes

### **Why Separate Backend/Frontend?**
1. **Scalability:** Scale backend and frontend independently
2. **Security:** Backend can be behind firewall, frontend exposed
3. **Flexibility:** Swap frontend (web, mobile app) without touching backend
4. **Professional:** Industry standard microservices architecture

### **Why Docker?**
1. **Consistency:** Same environment everywhere (dev, staging, production)
2. **Easy deployment:** One command to deploy anywhere
3. **Isolation:** Each service runs in its own container
4. **Portability:** Deploy to any cloud provider

---

## 🎯 Next Steps

- [ ] Deploy to Oracle Cloud
- [ ] Set up CI/CD with GitHub Actions
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Implement model versioning
- [ ] Add authentication/API keys
- [ ] Set up logging aggregation
- [ ] Add integration tests
- [ ] Create mobile-friendly frontend

---

## 📚 Documentation

- **Housing Deployment:** See `housing-deployment/DEPLOYMENT_GUIDE.md`
- **API Documentation:** 
  - Housing: http://localhost:8000/docs
  - Electricity: http://localhost:8002/docs

---

## 👥 Team

**Group:** Error400
- **Hamid Iqbal**
- **Ibrahim Afkir**

**Institution:** Thomas More  
**Course:** Cloud & AI  
**Date:** November 2025

---

## 📄 License

Educational project for Thomas More University College.

---

## 🙏 Acknowledgments

- UK Land Registry (Housing data)
- UK National Grid (Electricity data)
- FastAPI & Streamlit communities
- LightGBM & Scikit-learn teams

---

**Built with ❤️ for Cloud & AI Course**
