# Split Deployment: Railway (Backends) + Streamlit Cloud (Frontend)

## Architecture

```
Streamlit Cloud (Frontend)
         ↓
    combined_app.py
         ↓
    calls APIs at:
         ↓
Railway Backend URLs
  ├── Housing API: https://your-app.railway.app (port 8000)
  └── Electricity API: https://your-app.railway.app (port 8002)
```

## Step 1: Deploy Backends to Railway

### Railway Project Setup:

Your current Railway deployment has everything mixed together. We need to:

1. **Keep current Railway deployment** - it will run both backends
2. **Add environment variables** for Railway to expose the services
3. **Get the public URLs** for each backend

### How to set this up:

1. Go to your Railway project dashboard
2. You'll need to create **2 separate services**:
   - Service 1: Housing Backend (housing-deployment/backend)
   - Service 2: Electricity Backend (electricity-deployment/backend)

3. Each service gets its own public domain

## Step 2: Deploy Frontend to Streamlit Cloud

Once you have the Railway URLs for both backends, we'll:

1. Create a new branch or separate repo for Streamlit Cloud
2. Update `combined_app.py` with the Railway backend URLs
3. Add Streamlit Cloud config files
4. Deploy on https://share.streamlit.io

## Current Status

✅ Code is ready
✅ Models are in GitHub (under 100MB)
⏳ Need Railway backend URLs
⏳ Need to configure Streamlit Cloud deployment

## Next Steps

**For you to do in Railway:**

1. Go to Railway dashboard → Your project
2. Create **two separate services** from your GitHub repo:
   - **Service 1 (Housing)**:
     - Root directory: `housing-deployment/backend`
     - Start command: `python -m uvicorn api:app --host 0.0.0.0 --port $PORT`
     - Generate domain → Get URL (e.g., `https://housing-production.up.railway.app`)
   
   - **Service 2 (Electricity)**:
     - Root directory: `electricity-deployment/backend`
     - Start command: `python -m uvicorn api:app --host 0.0.0.0 --port $PORT`
     - Generate domain → Get URL (e.g., `https://electricity-production.up.railway.app`)

**Then tell me the two URLs and I'll:**
- Create a Streamlit Cloud version of `combined_app.py`
- Add deployment configs
- Guide you through Streamlit Cloud setup

## Why This Works

✅ **Railway strengths**: Runs FastAPI backends perfectly, handles multiple services
✅ **Streamlit Cloud strengths**: Free, fast, perfect for Streamlit apps
✅ **No credit card needed** for either platform
✅ **Both have free tiers** sufficient for your project
