# pink-cafe-app
Python application for the Pink Cafe Bristol project
# ☕ Pink Cafe App

### Machine Learning Dashboard for Sales Forecasting & Waste Reduction

---

## 📊 Project Overview

The **Pink Cafe App** is an interactive data analytics and machine learning dashboard developed using **Streamlit**. It is designed to analyse historical bakery sales data and forecast future demand for products such as coffee and croissants.

This system helps improve decision-making by reducing uncertainty in demand prediction and minimizing food waste.

---

## 🎯 Objectives

* Analyse historical sales data
* Identify top-performing products
* Visualise sales trends
* Forecast demand for the next **4 weeks**
* Compare forecasting models
* Support inventory and production planning

---

## 🚀 Key Features

* 📂 Upload and process CSV datasets
* 📈 Interactive dashboards using Plotly
* ☕ Coffee sales analysis (Americano, Cappuccino)
* 🥐 Croissant sales insights
* 🤖 Machine Learning model (Gradient Boosting)
* 📊 Time Series forecasting (SARIMA)
* 📉 Model evaluation metrics (MAE, RMSE, MAPE)
* ⚙️ Adjustable training window

---

## 🖼️ Dashboard Screenshots

### 🏠 Main Dashboard

![Main Dashboard](images/dashboard.png)

### 📈 Sales Trends

![Sales Trends](images/trends.png)

### 🤖 Forecast Results

![Forecast](images/forecast.png)

### 📊 Model Evaluation

![Model Evaluation](images/model.png)

---

## 🛠️ Technology Stack

* **Python**
* **Streamlit**
* **Pandas & NumPy**
* **Plotly**
* **Scikit-learn**
* **Statsmodels**

---

## 🧠 Forecasting Models

### 🔹 SARIMA (Statistical Model)

* Captures seasonal patterns
* Suitable for structured time-series data

### 🔹 Gradient Boosting (Machine Learning)

* Handles complex and non-linear relationships
* Achieved higher accuracy in this project

---

## 📂 Project Structure

```
Pink-Cafe-App/
│── app.py
│── data/
│── images/
│    ├── dashboard.png
│    ├── trends.png
│    ├── forecast.png
│    ├── model.png
│── README.md
│── requirements.txt
```

---

## ▶️ Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/azanmian676-lab/Pink-Cafe-App.git
cd Pink-Cafe-App
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Application

```
streamlit run app.py
```

### 4. Open in Browser

```
http://localhost:8501
```

---

## 📊 Dataset Details

* 📅 Period: March 2025 – October 2025
* 📦 Records: 690 daily entries
* ☕ Products: Americano, Cappuccino
* 🥐 Product: Croissant

---

## 📈 Results & Insights

* Sales trends show consistent coffee demand
* Croissant demand varies more frequently
* Gradient Boosting outperformed SARIMA
* Forecasting improves planning accuracy

---

## 🎯 Business Impact

* 📉 Reduced food waste
* 📦 Improved inventory control
* 📊 Better demand forecasting
* 🧠 Data-driven decision making

---

## 👨‍💻 Author

**Azan**
GitHub: https://github.com/azanmian676-lab

---

## 📌 Notes

* Developed as an academic project
* Designed for bakery demand forecasting
* Demonstrates practical use of ML in business

---

## ⭐ Acknowledgements

* Streamlit for dashboard framework
* Scikit-learn & Statsmodels for ML models
* Plotly for visualisation tools
