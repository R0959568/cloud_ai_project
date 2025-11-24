# 🚀 START HERE - Quick Deployment Guide

## ⚡ **Fastest Way to Test (No Docker Required)**

### **Option 1: Start Housing System** 🏠

```bash
cd /Users/hamidiqbal/Documents/ThomasMore/ML/cloud/data1_data2/cloud_ai_project-main
./start-housing.sh
```

Then open: **http://localhost:8501**

---

### **Option 2: Start Electricity System** ⚡

```bash
cd /Users/hamidiqbal/Documents/ThomasMore/ML/cloud/data1_data2/cloud_ai_project-main
./start-electricity.sh
```

Then open: **http://localhost:8502**

---

### **Option 3: Start Both Systems**

```bash
# Terminal 1 - Housing
./start-housing.sh

# Terminal 2 - Electricity  
./start-electricity.sh
```

Access:
- **Housing:** http://localhost:8501
- **Electricity:** http://localhost:8502

---

## 🛑 **Stop All Services**

```bash
./stop-all.sh
```

---

## 📋 **What Each Script Does**

### **start-housing.sh**
- Checks if model file exists
- Starts FastAPI backend on port 8000
- Starts Streamlit frontend on port 8501
- Shows you PIDs and log file locations

### **start-electricity.sh**
- Checks if model file exists
- Starts FastAPI backend on port 8002
- Starts Streamlit frontend on port 8502
- Shows you PIDs and log file locations

### **stop-all.sh**
- Stops all running backends and frontends
- Cleans up log files

---

## 🔍 **Troubleshooting**

### **"Address already in use" error**

Stop all services first:
```bash
./stop-all.sh
```

Then try starting again.

### **Check if services are running**

```bash
# Check ports
lsof -i :8000  # Housing backend
lsof -i :8501  # Housing frontend
lsof -i :8002  # Electricity backend
lsof -i :8502  # Electricity frontend
```

### **View logs**

```bash
# Housing logs
tail -f /tmp/housing_backend.log
tail -f /tmp/housing_frontend.log

# Electricity logs
tail -f /tmp/electricity_backend.log
tail -f /tmp/electricity_frontend.log
```

### **Model not found error**

Make sure models exist:
```bash
# Housing model
ls -lh models/lightgbm_housing.pkl

# Electricity model
ls -lh dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl
```

---

## 📦 **Installation Requirements**

Both systems need these Python packages:

```bash
pip3 install fastapi uvicorn streamlit pandas numpy joblib scikit-learn pyarrow requests plotly
```

Or install from requirements files:
```bash
# Housing
pip3 install -r housing-deployment/backend/requirements.txt
pip3 install -r housing-deployment/frontend/requirements.txt

# Electricity
pip3 install -r electricity-deployment/backend/requirements.txt
pip3 install -r electricity-deployment/frontend/requirements.txt
```

---

## ✅ **What to Test**

### **Housing System (http://localhost:8501)**
1. Select location (County, District, Town)
2. Choose property type (Detached, Semi-Detached, etc.)
3. Select tenure type (Freehold/Leasehold)
4. Pick year and month
5. Click "Predict Price"
6. See predicted price with range

### **Electricity System (http://localhost:8502)**
1. Select date and time
2. See derived features (day of week, season, etc.)
3. Click "Predict Demand"
4. See predicted electricity demand in MW
5. View which features were used

---

## 🌐 **API Documentation**

While services are running, check the auto-generated API docs:

- **Housing API:** http://localhost:8000/docs
- **Electricity API:** http://localhost:8002/docs

You can test the APIs directly from these pages!

---

## 📊 **What's Included**

### **Housing Prediction** 🏠
- **Model:** LightGBM
- **Data:** 994K records (1995-2017)
- **R² Score:** 0.67
- **Features:** Location, property type, tenure, date

### **Electricity Prediction** ⚡
- **Model:** Gradient Boosting
- **Data:** 25 years (2001-2025)
- **R² Score:** 0.70
- **Features:** 39 enhanced features (lag, rolling, temporal)

---

## 🎯 **For Your Presentation**

1. **Start both systems** before your presentation
2. **Test them** to make sure they work
3. **Keep terminal windows open** to show logs if needed
4. **Demonstrate predictions** on both systems
5. **Show API documentation** (the `/docs` endpoints)

---

## 📚 **More Documentation**

- **Complete Overview:** `README.md`
- **Deployment Ready Guide:** `DEPLOYMENT_READY.md`
- **Housing Details:** `housing-deployment/DEPLOYMENT_GUIDE.md`
- **Electricity Details:** `electricity-deployment/DEPLOYMENT_GUIDE.md`

---

## 🆘 **Quick Help**

```bash
# Start housing
./start-housing.sh

# Start electricity
./start-electricity.sh

# Stop everything
./stop-all.sh

# Check what's running
./quick-test-all.sh

# View logs
tail -f /tmp/housing_backend.log
tail -f /tmp/electricity_backend.log
```

---

**Ready to test? Run `./start-housing.sh` or `./start-electricity.sh` now!** 🚀
