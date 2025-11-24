# UK Housing Price Predictor - Streamlit App

A web application that predicts UK housing prices using a trained LightGBM model.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. File Structure

Make sure you have this structure:

```
your-project/
├── app.py                          # Streamlit app (this file)
├── requirements.txt                # Dependencies
├── models/
│   └── lightgbm_housing.pkl       # Your trained model
└── data/
    └── cleaned/
        └── housing_FULL_clean.parquet  # Your data
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 What You Need

1. **Trained Model**: `models/lightgbm_housing.pkl`
   - Generated from your LightGBM notebook
   - Make sure you ran the "Save Model" cell

2. **Data File**: `data/cleaned/housing_FULL_clean.parquet`
   - Used to populate dropdown options
   - Contains valid towns, districts, counties

## 🎯 How It Works

1. User selects property features:
   - Location (county, district, town)
   - Property type (Detached, Semi-detached, etc.)
   - Tenure (Freehold, Leasehold)
   - New build status
   - Year and month

2. Clicks "Predict Price"

3. App displays:
   - Predicted price
   - Price range (±10%)
   - Market comparison with similar properties

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy!

**Note:** Streamlit Cloud has file size limits (200MB). If your data is too large, consider:
- Using a smaller sample for dropdowns
- Storing data elsewhere (S3, database)

### Option 2: Local/Home Server

Run on your machine or Raspberry Pi:
```bash
streamlit run app.py --server.port 8501
```

### Option 3: Cloud VM (Oracle, AWS, etc.)

1. Create a VM (Oracle has free tier)
2. Install Python and dependencies
3. Run with nohup:
```bash
nohup streamlit run app.py --server.port 8501 &
```
4. Configure firewall to allow port 8501

### Option 4: Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## 🔧 Customization

### Change Model
Edit line 24 in `app.py`:
```python
with open('models/YOUR_MODEL.pkl', 'rb') as f:
```

### Change Data Path
Edit line 34 in `app.py`:
```python
df = pd.read_parquet('path/to/your/data.parquet')
```

### Update Model Info
Edit sidebar metrics (lines 50-53):
```python
st.sidebar.metric("R² Score", "YOUR_R2_SCORE")
```

## ⚠️ Troubleshooting

### "Model file not found"
- Make sure `models/lightgbm_housing.pkl` exists
- Check the path is correct relative to `app.py`

### "Data file not found"
- Ensure `data/cleaned/housing_FULL_clean.parquet` exists
- Or update the path in `load_data()` function

### Categories Not Found
- Your model must be trained with the same features
- Column names must match exactly
- Categorical dtype must be set for LightGBM

### File Too Large for GitHub
- Add to `.gitignore`:
```
models/*.pkl
data/*.parquet
```
- Store files separately (Google Drive, S3, etc.)
- Download them during deployment

## 📊 Model Performance

- **Model Type:** LightGBM Regressor
- **Training Data:** 5.9M transactions (1995-2017)
- **R² Score:** ~67%
- **Features:** 9 (location, property type, tenure, date)

## 👥 Team

[Add your team name and members here]

## 📝 License

[Add your license here]
