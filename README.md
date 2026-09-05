Dynamic Pricing Optimization Engine

An AI-powered Dynamic Pricing Optimization Engine that combines LightGBM demand forecasting, time-series signals, price elasticity, and RL-style price optimization to recommend the optimal product price for the next day.

🚀 Project Overview

The system predicts next-day product demand and evaluates multiple possible price changes to identify the pricing action that maximizes expected revenue while considering demand response and price movement penalties.

Key capabilities
📈 Next-day demand forecasting
🤖 LightGBM-based demand prediction
🕒 Time-series demand and price signals
💰 Price elasticity analysis
🧮 RL-style pricing optimization
📊 Candidate price evaluation
🇮🇳 INR-based pricing
🎯 Product and date selection
📋 Pricing recommendation dashboard
🌑 Professional dark-themed Streamlit interface
🧠 How It Works
Historical Sales Data
        ↓
Data Processing & Feature Engineering
        ↓
Time-Series Features
        ↓
LightGBM Demand Forecast
        ↓
Price Elasticity
        ↓
Candidate Price Actions
        ↓
Expected Demand & Revenue
        ↓
RL-Style Reward Optimization
        ↓
Recommended Price

The forecasting component creates lag, rolling, calendar, historical-product and price-related features before generating the next-day demand prediction.

The pricing optimizer evaluates candidate price changes from -10% to +10% and selects the action with the highest reward.

📊 Features Used

The model uses features including:

Demand
Demand lags: 1, 7, 14 and 28 days
Rolling demand means
Rolling demand standard deviations
Demand trends
Average price
Price lag
Price change percentage
Rolling price mean
Revenue
Transaction count
Day of week
Week of year
Month
Quarter
Weekend indicator
Product age
Historical demand statistics
Historical non-zero demand rate

These are defined in the application's V3 feature list.

💡 Pricing Optimization

The optimizer tests several possible pricing actions:

-10.0%
-7.5%
-5.0%
-2.5%
 0.0%
+2.5%
+5.0%
+7.5%
+10.0%

For every candidate price, the system estimates:

Expected demand
Expected revenue
Price movement penalty
Demand loss penalty
Overall reward

The highest-reward candidate becomes the recommended pricing action.

🖥️ Dashboard

The Streamlit application provides:

Product selection
Analysis date selection
Demand forecast
Current price
Recommended price
Price change
Expected demand
Expected revenue
Optimization results
Pricing actions
Model information

The interface is designed with a dark navy/purple professional theme and a portfolio-ready dashboard layout.

🛠️ Tech Stack
Technology	Purpose
Python	Core development
Pandas	Data processing
NumPy	Numerical computation
LightGBM	Demand forecasting
Plotly	Interactive visualizations
Scikit-learn	ML utilities
Streamlit	Web application
Pickle	Model loading

For deployment, a smaller deployment dataset can be used instead of committing the original large forecasting dataset to GitHub.

Run the application:

streamlit run app.py

The application will open in your browser.

📦 Requirements
streamlit
pandas
numpy
plotly
scikit-learn
lightgbm
joblib
☁️ Streamlit Community Cloud

The project can be deployed using Streamlit Community Cloud.

Make sure the GitHub repository contains:

app.py
requirements.txt
models/
data/processed/
reports/

The application expects the forecasting dataset under:

data/processed/forecast_model_data_deploy.csv

The application constructs its expected time-series features after loading the dataset.

🎯 Objective

The main objective is to build an intelligent pricing system that moves beyond static pricing by combining demand forecasting, time-series information, customer price sensitivity, and optimization to support data-driven pricing decisions.

🔮 Future Enhancements
Real-time sales data integration
Reinforcement learning environment with continuous price actions
Competitor price signals
Inventory-aware pricing
Promotion optimization
Multi-product optimization
Online model retraining
Automated model monitoring

👩‍💻 Author
Shalini G

Dynamic Pricing Optimization Engine — AI-driven demand forecasting and intelligent pricing recommendations.
