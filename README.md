# Airbnb Dynamic Pricing Recommendation Engine

---

## 📘 Project Overview
This project develops a Dynamic Pricing Recommendation Engine for Airbnb listings. It predicts optimal nightly prices based on features such as property type, number of bedrooms, amenities, neighbourhood, and host characteristics. The main goal is to help Airbnb hosts set competitive and profitable prices using machine learning, data analysis, and visualization.

---

## 🎯 Objectives
- Predict nightly Airbnb prices using regression-based machine learning models.  
- Identify key factors influencing pricing such as amenities, property type, and neighbourhood.  
- Build an interactive Tableau dashboard to visualize insights.  
- Develop a recommendation mechanism to dynamically suggest optimal prices.  
- Deliver a complete end-to-end workflow from raw data to model recommendations and dashboard visualization.

---

## ⚙️ Tools & Technologies
- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, lightgbm, shap, joblib, category_encoders, matplotlib, seaborn  
- **Visualization:** Tableau Public  
- **IDE / Notebook:** JupyterLab or VS Code  
- **Optional Deployment:** Flask (for API service)

---

## 📂 Project Structure
airbnb_pricing/  
├─ data/  
│  ├─ Airbnb_Data.csv  
│  ├─ airbnb_raw_cleaned.csv  
│  ├─ airbnb_processed.csv  
│  └─ airbnb_features.csv  
│  
├─ src/  
│  ├─ data_ingest.py  
│  ├─ data_cleaning.py  
│  ├─ features.py  
│  ├─ train_model.py  
│  ├─ predict.py  
│  └─ serve_api.py  
│  
├─ models/  
│  ├─ model_lightgbm.joblib  
│  └─ model_ridge.joblib  
│  
├─ notebooks/  
│  ├─ 01_EDA.ipynb  
│  └─ 02_explainability.ipynb  
│  
├─ dashboard/  
│  ├─ airbnb_dashboard.twbx  
│  └─ airbnb_dashboard_screenshot.png  
│  
├─ reports/  
│  └─ Airbnb_Dynamic_Pricing_Recommendation_Engine_Report_Final.pdf  
│  
├─ requirements.txt  
└─ README.md  

---

## 🚀 How to Run the Project

### Step 1: Create and Activate Virtual Environment
Use Anaconda Prompt or terminal and run:
```bash
conda create -n airbnb_env python=3.9 -y
conda activate airbnb_env