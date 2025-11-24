
# 🚀 Complete Deployment Guide

## Architecture

```
GitHub Push → GitHub Actions → Auto Deploy
     ↓
Frontend (Streamlit:8501) → Backend API (FastAPI:8000) → LightGBM Model
```

---

## 📁 Project Structure

```
your-project/
├── backend/
│   ├── api.py                    # FastAPI backend
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                    # Streamlit frontend
│   ├── requirements.txt
│   └── Dockerfile
├── models/
│   └── lightgbm_housing.pkl      # Your trained model
├── data/
│   └── cleaned/
│       └── housing_FULL_clean.parquet
├── .github/
│   └── workflows/
│       └── deploy.yml            # Auto-deployment pipeline
├── docker-compose.yml
└── README.md
```

---

## 🧪 TEST LOCALLY FIRST

### Step 1: Setup

```bash
# Put your files in the right places
cp path/to/lightgbm_housing.pkl models/
cp path/to/housing_FULL_clean.parquet data/cleaned/
```

### Step 2: Test Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload
```

Open http://localhost:8000/docs to see API documentation

### Step 3: Test Frontend (New Terminal)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

**Make sure frontend can call backend!**

### Step 4: Test with Docker (Optional but recommended)

```bash
# From project root
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## ☁️ DEPLOY TO ORACLE CLOUD (FREE)

### Step 1: Create Oracle Account

1. Go to [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Sign up (requires credit card but won't charge)
3. Choose region close to you

### Step 2: Create VM Instance

1. Go to **Compute** → **Instances**
2. Click **Create Instance**
3. Choose:
   - **Image:** Ubuntu 22.04
   - **Shape:** VM.Standard.E2.1.Micro (Always Free)
   - **Boot Volume:** 50GB
4. Download SSH keys
5. Note the **Public IP**

### Step 3: Configure Firewall

1. Go to **Networking** → **Virtual Cloud Networks**
2. Click your VCN → Security Lists
3. Add Ingress Rules:
   - Port 8000 (Backend)
   - Port 8501 (Frontend)
   - Source: 0.0.0.0/0

### Step 4: Setup Server

```bash
# SSH into your server
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt install docker-compose -y

# Create project directory
mkdir -p /opt/housing-predictor
cd /opt/housing-predictor
```

### Step 5: Upload Your Files

**From your local machine:**

```bash
# Upload project files
scp -i your-key.pem -r backend frontend docker-compose.yml \
    ubuntu@YOUR_PUBLIC_IP:/opt/housing-predictor/

# Upload model (if small enough)
scp -i your-key.pem -r models \
    ubuntu@YOUR_PUBLIC_IP:/opt/housing-predictor/

# Upload data sample (for dropdowns)
scp -i your-key.pem -r data \
    ubuntu@YOUR_PUBLIC_IP:/opt/housing-predictor/
```

**If files are too large:**
- Upload to Google Drive/Dropbox
- Download on server: `wget YOUR_LINK`

### Step 6: Start Services

```bash
# SSH back into server
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

cd /opt/housing-predictor

# Start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Step 7: Test Deployment

- Backend API: `http://YOUR_PUBLIC_IP:8000/docs`
- Frontend: `http://YOUR_PUBLIC_IP:8501`

---

## 🔄 GITHUB ACTIONS PIPELINE

### Step 1: Setup Secrets

Go to your GitHub repo → Settings → Secrets → Actions

Add these secrets:

1. **SSH_PRIVATE_KEY**
   ```bash
   # Your Oracle VM SSH private key content
   cat your-key.pem
   ```

2. **SERVER_HOST**
   ```
   YOUR_PUBLIC_IP
   ```

3. **SERVER_USER**
   ```
   ubuntu
   ```

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial deployment setup"
git branch -M main
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

### Step 3: Auto-Deployment

Now every time you push to `main`:
1. GitHub Actions builds Docker images
2. Uploads to your Oracle VM
3. Restarts services automatically

---

## 🔧 TROUBLESHOOTING

### Backend won't start
```bash
# Check logs
docker logs housing-backend

# Common issue: Model file missing
# Make sure models/lightgbm_housing.pkl exists
```

### Frontend can't reach backend
```bash
# Update frontend/.streamlit/secrets.toml
API_URL = "http://YOUR_PUBLIC_IP:8000"

# Or set environment variable in docker-compose.yml
```

### Port already in use
```bash
# Stop existing containers
docker-compose down

# Change ports in docker-compose.yml if needed
```

### Model too large for GitHub
```bash
# Add to .gitignore
echo "models/*.pkl" >> .gitignore
echo "data/*.parquet" >> .gitignore

# Upload separately to server
```

---

## 📊 WHAT YOU'VE BUILT

✅ **Frontend:** Streamlit web interface  
✅ **Backend:** FastAPI REST API  
✅ **Model:** LightGBM serving predictions  
✅ **Pipeline:** GitHub Actions auto-deployment  
✅ **Hosting:** Oracle Cloud (free tier)  
✅ **Docker:** Containerized services  

---

## 🎓 FOR YOUR PRESENTATION

**What to show:**

1. **Architecture diagram** (frontend → backend → model)
2. **Live demo** (your deployed URL)
3. **GitHub Actions** (show pipeline running)
4. **Docker Compose** (explain why separation matters)

**Key points:**

- "We separated frontend and backend like real companies do"
- "We automated deployment with GitHub Actions"
- "Every code push triggers auto-deployment"
- "Frontend calls backend via REST API"
- "Everything runs in Docker containers"

---

## 🆘 QUICK COMMANDS

```bash
# Local testing
cd backend && uvicorn api:app --reload  # Terminal 1
cd frontend && streamlit run app.py      # Terminal 2

# Docker testing
docker-compose up --build

# Deploy to server
git push origin main  # Auto-deploys via GitHub Actions

# Manual deploy
scp -r * ubuntu@SERVER:/opt/housing-predictor/
ssh ubuntu@SERVER "cd /opt/housing-predictor && docker-compose up -d"

# Check logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart
docker-compose restart
```

---

## 💡 TIPS

1. **Test locally first** - Don't push broken code
2. **Keep model files small** - Use .gitignore for large files
3. **Document everything** - Your future self will thank you
4. **Use meaningful commits** - "Fix API endpoint" not "stuff"
5. **Monitor logs** - `docker-compose logs -f` is your friend

---

## 📝 TODO CHECKLIST

- [ ] Test backend locally (http://localhost:8000/docs)
- [ ] Test frontend locally (http://localhost:8501)
- [ ] Test with Docker Compose
- [ ] Create Oracle VM
- [ ] Configure firewall rules
- [ ] Upload files to server
- [ ] Test deployment (http://PUBLIC_IP:8501)
- [ ] Setup GitHub secrets
- [ ] Push to GitHub
- [ ] Verify GitHub Actions works
- [ ] Prepare presentation demo

---

**Good luck! 🚀**$
=======
# 🚀 Complete Deployment Guide

## Architecture

```
GitHub Push → GitHub Actions → Auto Deploy
     ↓
Frontend (Streamlit:8501) → Backend API (FastAPI:8000) → LightGBM Model
```

---

## 📁 Project Structure

```
your-project/
├── backend/
│   ├── api.py                    # FastAPI backend
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                    # Streamlit frontend
│   ├── requirements.txt
│   └── Dockerfile
├── models/
│   └── lightgbm_housing.pkl      # Your trained model
├── data/
│   └── cleaned/
│       └── housing_FULL_clean.parquet
├── .github/
│   └── workflows/
│       └── deploy.yml            # Auto-deployment pipeline
├── docker-compose.yml
└── README.md
```

---

## 🧪 TEST LOCALLY FIRST

### Step 1: Setup

```bash
# Put your files in the right places
cp path/to/lightgbm_housing.pkl models/
cp path/to/housing_FULL_clean.parquet data/cleaned/
```

### Step 2: Test Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload
```

Open http://localhost:8000/docs to see API documentation

### Step 3: Test Frontend (New Terminal)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

**Make sure frontend can call backend!**

### Step 4: Test with Docker (Optional but recommended)

```bash
# From project root
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## ☁️ DEPLOY TO ORACLE CLOUD (FREE)

### Step 1: Create Oracle Account

1. Go to [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Sign up (requires credit card but won't charge)
3. Choose region close to you

### Step 2: Create VM Instance

1. Go to **Compute** → **Instances**
2. Click **Create Instance**
3. Choose:
   - **Image:** Ubuntu 22.04
   - **Shape:** VM.Standard.E2.1.Micro (Always Free)
   - **Boot Volume:** 50GB
4. Download SSH keys
5. Note the **Public IP**

### Step 3: Configure Firewall

1. Go to **Networking** → **Virtual Cloud Networks**
2. Click your VCN → Security Lists
3. Add Ingress Rules:
   - Port 8000 (Backend)
   - Port 8501 (Frontend)
   - Source: 0.0.0.0/0

### Step 4: Setup Server

```bash
# SSH into your server
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt install docker-compose -y

# Create project directory
mkdir -p /opt/housing-predictor
cd /opt/housing-predictor
```

### Step 5: Upload Your Files

**From your local machine:**

```bash
# Upload project files
scp -i your-key.pem -r backend frontend docker-compose.yml \
    ubuntu@YOUR_PUBLIC_IP:/opt/housing-predictor/

# Upload model (if small enough)
scp -i your-key.pem -r models \
    ubuntu@YOUR_PUBLIC_IP:/opt/housing-predictor/

# Upload data sample (for dropdowns)
scp -i your-key.pem -r data \
    ubuntu@YOUR_PUBLIC_IP:/opt/housing-predictor/
```

**If files are too large:**
- Upload to Google Drive/Dropbox
- Download on server: `wget YOUR_LINK`

### Step 6: Start Services

```bash
# SSH back into server
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP

cd /opt/housing-predictor

# Start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Step 7: Test Deployment

- Backend API: `http://YOUR_PUBLIC_IP:8000/docs`
- Frontend: `http://YOUR_PUBLIC_IP:8501`

---

## 🔄 GITHUB ACTIONS PIPELINE

### Step 1: Setup Secrets

Go to your GitHub repo → Settings → Secrets → Actions

Add these secrets:

1. **SSH_PRIVATE_KEY**
   ```bash
   # Your Oracle VM SSH private key content
   cat your-key.pem
   ```

2. **SERVER_HOST**
   ```
   YOUR_PUBLIC_IP
   ```

3. **SERVER_USER**
   ```
   ubuntu
   ```

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial deployment setup"
git branch -M main
git remote add origin YOUR_GITHUB_REPO
git push -u origin main
```

### Step 3: Auto-Deployment

Now every time you push to `main`:
1. GitHub Actions builds Docker images
2. Uploads to your Oracle VM
3. Restarts services automatically

---

## 🔧 TROUBLESHOOTING

### Backend won't start
```bash
# Check logs
docker logs housing-backend

# Common issue: Model file missing
# Make sure models/lightgbm_housing.pkl exists
```

### Frontend can't reach backend
```bash
# Update frontend/.streamlit/secrets.toml
API_URL = "http://YOUR_PUBLIC_IP:8000"

# Or set environment variable in docker-compose.yml
```

### Port already in use
```bash
# Stop existing containers
docker-compose down

# Change ports in docker-compose.yml if needed
```

### Model too large for GitHub
```bash
# Add to .gitignore
echo "models/*.pkl" >> .gitignore
echo "data/*.parquet" >> .gitignore

# Upload separately to server
```

---

## 📊 WHAT YOU'VE BUILT

✅ **Frontend:** Streamlit web interface  
✅ **Backend:** FastAPI REST API  
✅ **Model:** LightGBM serving predictions  
✅ **Pipeline:** GitHub Actions auto-deployment  
✅ **Hosting:** Oracle Cloud (free tier)  
✅ **Docker:** Containerized services  

---

## 🎓 FOR YOUR PRESENTATION

**What to show:**

1. **Architecture diagram** (frontend → backend → model)
2. **Live demo** (your deployed URL)
3. **GitHub Actions** (show pipeline running)
4. **Docker Compose** (explain why separation matters)

**Key points:**

- "We separated frontend and backend like real companies do"
- "We automated deployment with GitHub Actions"
- "Every code push triggers auto-deployment"
- "Frontend calls backend via REST API"
- "Everything runs in Docker containers"

---

## 🆘 QUICK COMMANDS

```bash
# Local testing
cd backend && uvicorn api:app --reload  # Terminal 1
cd frontend && streamlit run app.py      # Terminal 2

# Docker testing
docker-compose up --build

# Deploy to server
git push origin main  # Auto-deploys via GitHub Actions

# Manual deploy
scp -r * ubuntu@SERVER:/opt/housing-predictor/
ssh ubuntu@SERVER "cd /opt/housing-predictor && docker-compose up -d"

# Check logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart
docker-compose restart
```

---

## 💡 TIPS

1. **Test locally first** - Don't push broken code
2. **Keep model files small** - Use .gitignore for large files
3. **Document everything** - Your future self will thank you
4. **Use meaningful commits** - "Fix API endpoint" not "stuff"
5. **Monitor logs** - `docker-compose logs -f` is your friend

---

## 📝 TODO CHECKLIST

- [ ] Test backend locally (http://localhost:8000/docs)
- [ ] Test frontend locally (http://localhost:8501)
- [ ] Test with Docker Compose
- [ ] Create Oracle VM
- [ ] Configure firewall rules
- [ ] Upload files to server
- [ ] Test deployment (http://PUBLIC_IP:8501)
- [ ] Setup GitHub secrets
- [ ] Push to GitHub
- [ ] Verify GitHub Actions works
- [ ] Prepare presentation demo

---

**Good luck! 🚀**

