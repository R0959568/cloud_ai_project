# 🚂 Railway Deployment Guide

## ✅ Files Ready for Deployment

All configuration files have been created:
- ✅ `railway.json` - Railway configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Start command
- ✅ `.railwayignore` - Exclude large CSV files

## 🚀 Deployment Steps

### Option 1: Deploy via GitHub (Recommended)

1. **Create GitHub Repository**
   ```bash
   cd /Users/hamidiqbal/Documents/ThomasMore/ML/cloud/data1_data2/cloud_ai_project-main
   git init
   git add .
   git commit -m "Initial commit - ML prediction system"
   ```

2. **Push to GitHub**
   - Create a new repo on GitHub.com
   - Follow the instructions to push your code

3. **Deploy on Railway**
   - Go to https://railway.app/dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect and deploy!

### Option 2: Deploy via Railway CLI

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize and Deploy**
   ```bash
   cd /Users/hamidiqbal/Documents/ThomasMore/ML/cloud/data1_data2/cloud_ai_project-main
   railway init
   railway up
   ```

4. **Get Your URL**
   ```bash
   railway domain
   ```

## ⚠️ Important Notes

### File Size Limits
Railway has upload limits. Your project excludes:
- ❌ Raw CSV files (too large)
- ✅ Model files (.pkl) - included
- ✅ Cleaned data (.parquet) - included

### Models & Data Included
These files WILL be deployed:
```
✅ models/lightgbm_housing.pkl
✅ models/gradient_boosting_enhanced.pkl
✅ data/cleaned/housing_clean.parquet
✅ dataset_2_electricity_app/data/interim/elec_cleaned_full_sample.csv
```

### Environment Variables
Railway will automatically set:
- `PORT` - The port your app runs on
- No other env vars needed!

## 🔧 After Deployment

1. **Your app will be live at**: `https://your-app-name.up.railway.app`

2. **Test the endpoints**:
   - Frontend: `https://your-app.up.railway.app`
   - Housing API: `https://your-app.up.railway.app:8000/docs`
   - Electricity API: `https://your-app.up.railway.app:8002/docs`

3. **Check logs**:
   ```bash
   railway logs
   ```

## 💰 Railway Free Tier

- **$5 credit/month** (resets monthly)
- **500 hours execution time**
- **100 GB bandwidth**
- **1 GB RAM, 1 vCPU**

Should be enough for your project!

## 🐛 Troubleshooting

### If deployment fails:

1. **Check logs**:
   ```bash
   railway logs
   ```

2. **Verify files are uploaded**:
   ```bash
   railway run ls -la
   ```

3. **Test locally first**:
   ```bash
   ./start-combined.sh
   ```

## 📊 What Gets Deployed

```
Your Railway App
├── Combined Frontend (Streamlit on $PORT)
├── Housing Backend (FastAPI on port 8000)
├── Electricity Backend (FastAPI on port 8002)
├── Models (lightgbm, gradient_boosting)
└── Data (parquet, csv files)
```

## 🎉 Success!

Once deployed, share your Railway URL with:
- ✅ Classmates
- ✅ Instructors
- ✅ Your portfolio
- ✅ Anywhere!

No credit card, no complex setup, just works! 🚀
