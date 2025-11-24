# ⚡ UK Electricity Demand Prediction - Deployment Guide

Complete deployment guide for the UK Electricity Demand Prediction System.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Oracle Cloud Deployment](#oracle-cloud-deployment)
5. [API Documentation](#api-documentation)
6. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

```
User Browser
     ↓
Streamlit Frontend (:8502)
     ↓ HTTP POST
FastAPI Backend (:8002)
     ↓
Gradient Boosting Model
     ↓
Prediction (MW)
```

**Key Features:**
- **Backend:** FastAPI REST API
- **Frontend:** Streamlit web interface
- **Model:** Gradient Boosting (39 features, R²=0.70)
- **Deployment:** Docker containers

---

## 💻 Local Development

### **Step 1: Test Backend**

```bash
cd electricity-deployment/backend

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn api:app --reload --port 8002
```

Access API docs: http://localhost:8002/docs

**Test with curl:**
```bash
curl http://localhost:8002/health

curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"prediction_datetime": "2024-06-15T14:30:00"}'
```

### **Step 2: Test Frontend**

```bash
cd electricity-deployment/frontend

# Install dependencies
pip install -r requirements.txt

# Update API_URL in app.py if needed
# API_URL = "http://localhost:8002"

# Start frontend
streamlit run app.py --server.port 8502
```

Access app: http://localhost:8502

---

## 🐳 Docker Deployment

### **Quick Start**

```bash
cd electricity-deployment

# Build and start services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

**Access:**
- Frontend: http://localhost:8502
- Backend API: http://localhost:8002/docs

### **Docker Commands**

```bash
# View logs
docker-compose logs -f

# View backend logs only
docker-compose logs -f backend

# Stop services
docker-compose down

# Restart
docker-compose restart

# Rebuild without cache
docker-compose build --no-cache
docker-compose up
```

---

## ☁️ Oracle Cloud Deployment

### **Step 1: Create VM Instance**

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Create Compute Instance:
   - **Image:** Ubuntu 22.04
   - **Shape:** VM.Standard.E2.1.Micro (Always Free)
   - **Boot Volume:** 50GB
3. Download SSH keys
4. Note the **Public IP**

### **Step 2: Configure Firewall**

**In Oracle Cloud Console:**
1. Go to **Networking** → **Virtual Cloud Networks**
2. Click your VCN → **Security Lists** → **Default Security List**
3. Click **Add Ingress Rules**

Add these rules:

| Source CIDR | IP Protocol | Source Port | Destination Port | Description |
|-------------|-------------|-------------|------------------|-------------|
| 0.0.0.0/0   | TCP         | All         | 8002             | Backend API |
| 0.0.0.0/0   | TCP         | All         | 8502             | Frontend    |

### **Step 3: Setup Server**

```bash
# SSH into VM
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt install docker-compose -y

# Logout and login again for group changes
exit
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
```

### **Step 4: Upload Project Files**

**Option A: Upload from local machine**

```bash
# From your local machine
cd /path/to/cloud_ai_project-main

# Upload electricity deployment
scp -i your-key.pem -r electricity-deployment ubuntu@YOUR_PUBLIC_IP:/home/ubuntu/

# Upload model files
scp -i your-key.pem -r dataset_2_electricity_app/data ubuntu@YOUR_PUBLIC_IP:/home/ubuntu/
```

**Option B: Clone from GitHub**

```bash
# On the VM
git clone YOUR_GITHUB_REPO_URL
cd cloud_ai_project-main
```

### **Step 5: Start Services**

```bash
# On the VM
cd electricity-deployment

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Verify services are running
docker ps
```

### **Step 6: Test Deployment**

```bash
# Test backend
curl http://YOUR_PUBLIC_IP:8002/health

# Test prediction
curl -X POST "http://YOUR_PUBLIC_IP:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"prediction_datetime": "2024-06-15T14:30:00"}'
```

**Access frontend:** http://YOUR_PUBLIC_IP:8502

---

## 📚 API Documentation

### **Endpoints**

#### **GET /** - Root
```bash
curl http://localhost:8002/
```
Response:
```json
{
  "status": "online",
  "message": "UK Electricity Demand Prediction API",
  "model_loaded": true,
  "historical_data_loaded": true
}
```

#### **GET /health** - Health Check
```bash
curl http://localhost:8002/health
```

#### **POST /predict** - Make Prediction
```bash
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_datetime": "2024-06-15T14:30:00"
  }'
```

Response:
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

#### **GET /model-info** - Model Information
```bash
curl http://localhost:8002/model-info
```

---

## 🔧 Troubleshooting

### **Backend won't start**

```bash
# Check logs
docker logs electricity-backend

# Common issue: Model file not found
# Verify model exists
ls -la ../dataset_2_electricity_app/data/final/models/gradient_boosting_enhanced.pkl

# Check volume mounts in docker-compose.yml
```

### **Frontend can't reach backend**

```bash
# Check backend is running
curl http://localhost:8002/health

# Update API_URL in frontend/app.py
# For Docker: API_URL = "http://backend:8000"
# For local: API_URL = "http://localhost:8002"

# Rebuild frontend
docker-compose build frontend
docker-compose up -d
```

### **Port already in use**

```bash
# Stop existing containers
docker-compose down

# Check what's using the port
sudo lsof -i :8002
sudo lsof -i :8502

# Kill the process or change ports in docker-compose.yml
```

### **Model predictions seem wrong**

```bash
# Check if historical data is loaded
curl http://localhost:8002/model-info | jq '.historical_data'

# Verify data file exists
ls -la ../dataset_2_electricity_app/data/interim/elec_cleaned_full.parquet
```

### **High memory usage**

The Gradient Boosting model and historical data (~25 years) can use significant memory.

**Solutions:**
- Increase VM memory
- Use model compression
- Sample historical data for lag features

---

## 📊 Model Performance

- **Model Type:** Gradient Boosting Regressor
- **R² Score:** 0.70
- **MAE:** 2,353.23 MW
- **RMSE:** 3,107.24 MW
- **Features:** 39 (lag, rolling, temporal, holidays)
- **Training Data:** 2001-2025 (25 years)

---

## 🎯 Production Checklist

- [ ] Backend starts successfully
- [ ] Frontend connects to backend
- [ ] Health endpoints return 200 OK
- [ ] Predictions are reasonable (20,000-45,000 MW)
- [ ] Docker containers auto-restart
- [ ] Firewall rules configured
- [ ] HTTPS/SSL configured (if needed)
- [ ] Monitoring setup
- [ ] Backups configured

---

## 🔐 Security Considerations

**For Production:**

1. **API Authentication:**
   - Add API keys
   - Implement JWT tokens

2. **HTTPS:**
   - Use Nginx reverse proxy
   - Get SSL certificate (Let's Encrypt)

3. **Firewall:**
   - Restrict source IPs if possible
   - Use security groups

4. **Rate Limiting:**
   - Prevent API abuse
   - Use middleware

---

## 📝 Maintenance

### **Update Model**

```bash
# Upload new model
scp -i key.pem new_model.pkl ubuntu@IP:/home/ubuntu/dataset_2_electricity_app/data/final/models/

# Restart backend
docker-compose restart backend
```

### **View Logs**

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f backend
```

### **Backup**

```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz \
  dataset_2_electricity_app/data/final/models/ \
  dataset_2_electricity_app/data/interim/
```

---

## 🆘 Support

**Issues?**

1. Check logs: `docker-compose logs -f`
2. Verify model files exist
3. Test backend separately: `curl http://localhost:8002/health`
4. Check firewall rules
5. Review this guide again

---

## 🎓 For Presentation

**Key Points to Demonstrate:**

1. **Architecture:** Show backend/frontend separation
2. **Live Demo:** Make predictions via web interface
3. **API Docs:** Show FastAPI automatic documentation
4. **Docker:** Explain containerization benefits
5. **Features:** Highlight 39 enhanced features (lag, rolling, etc.)
6. **Performance:** R²=0.70, reasonable predictions

---

**Good luck with your deployment! ⚡🚀**
