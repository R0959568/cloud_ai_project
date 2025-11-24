# 🏠 UK Housing Price Predictor

Machine learning project predicting UK housing prices with automated deployment pipeline.

## 📊 Project Overview

- **Dataset:** 5.9M UK housing transactions (1995-2017)
- **Model:** LightGBM Regressor (R² ~67%)
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Deployment:** Docker + GitHub Actions → Oracle Cloud

## 🚀 Quick Start

### Local Development

1. **Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload
```
Access API docs: http://localhost:8000/docs

2. **Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Access app: http://localhost:8501

### Docker

```bash
docker-compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:8501

## 📁 Project Structure

```
.
├── backend/              # FastAPI backend
│   ├── api.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # Streamlit frontend
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── models/               # Trained models
│   └── lightgbm_housing.pkl
├── data/                 # Data files
│   └── cleaned/
│       └── housing_FULL_clean.parquet
├── .github/
│   └── workflows/
│       └── deploy.yml    # CI/CD pipeline
├── docker-compose.yml
└── DEPLOYMENT_GUIDE.md   # Detailed deployment instructions
```

## 🔧 Technologies

- **ML:** LightGBM, Scikit-learn, Pandas
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **DevOps:** Docker, GitHub Actions
- **Hosting:** Oracle Cloud (free tier)

## 🌐 Live Demo

[Add your deployed URL here]

## 👥 Team

[Add your team name and members]

## 📖 Documentation

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete deployment instructions.

## 📝 License

[Add your license]
