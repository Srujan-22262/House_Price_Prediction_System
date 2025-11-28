🏠 Bengaluru House Price Prediction
📌 An AI-powered real-estate price estimator built using Machine Learning & Streamlit
🚀 Project Overview

This project predicts house prices in Bengaluru using Machine Learning.
It includes:

✔ Data cleaning & preprocessing

✔ Feature engineering (BHK calculation, price per sqft, outlier removal)

✔ One-hot encoding for categorical features

✔ Model training & evaluation

✔ Exporting the best ML model

✔ A beautiful Streamlit web app

✔ Interactive charts, insights & detailed price explanation

This system helps users explore how size, BHK, location, and area type impact the final price.


🧹 Dataset Cleaning & Processing
Applied transformations:

Remove missing or incorrect values

Convert total_sqft to numeric

Create new features:

bhk

price_per_sqft

Remove outliers using:

IQR (sqft/bhk)

Standard deviation (price/sqft)

One-hot encode:

location

area_type

availability (only "Ready To Move" used in app)

Final dataset shape:

✔ ~325 features after one-hot encoding
✔ Numeric + categorical converted to ML-ready format


🧠 Machine Learning Model :
Models tested:

Linear Regression

Decision Tree

Random Forest

Gradient Boosting


Best performing model:

⭐⭐ Linear Regression 


🖥 Streamlit Web App
Features:

Beautiful UI (CSS customized)

Sidebar-driven dynamic inputs:

Total Sqft

Bathrooms

Balconies

BHK

Location

Area Type

Availability (Fixed to Ready To Move)

Real-time prediction

Price per Sqft, Total Value & Summary cards

Interactive charts using Plotly:

BHK vs Price comparison

Area vs Price trend



Tech used:

Streamlit

Numpy

Pandas

Plotly

Scikit-learn

Pickle



▶ How to Run Locally
1️⃣ Clone the repo
git clone https://github.com/Srujan-22262/bengaluru-house-price-prediction
cd bengaluru-house-price-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py


Your app will open in the browser at:

http://localhost:8501
