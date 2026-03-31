# ☕ Pink Cafe App

### Bakery Sales Prediction Dashboard (Streamlit + Machine Learning)

---

## 📊 Project Overview

The **Pink Cafe App** is an interactive Streamlit dashboard developed to analyse bakery sales data and forecast future demand.

It supports:

* 📈 Sales trend analysis
* 🤖 Machine learning forecasting
* 📉 Model evaluation

The system helps improve **inventory planning** and reduce **food waste**.

---

## 🚀 Key Features

* Upload coffee & croissant datasets
* Automatic data preprocessing
* Identify top-selling products
* Visualise last 4 weeks of sales
* Forecast next 28 days
* Compare forecasting models
* Evaluation metrics (MAE, RMSE, MAPE)
* Zoom into short forecast windows

---

## 🖼️ Dashboard Screenshots

### 🏠 Main Dashboard

![Dashboard](images/dashboard.png)

### 📈 Sales Trends

![Trends](images/trends.png)

### 🤖 Forecast Output

![Forecast](images/forecast.png)

### 📊 Model Evaluation

![Model](images/model.png)

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas & NumPy
* Plotly
* Scikit-learn
* Statsmodels

---

## ▶️ How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the app

```
streamlit run app.py
```

### 3. Open in browser

```
http://localhost:8501
```

---

## 📂 Required Input Files

⚠️ You MUST upload BOTH files in the app:

### ☕ Coffee CSV

* Must contain **Date column**
* First row = coffee product names
* Remaining columns = sales values

### 🥐 Croissant CSV

* Must contain:

  * Date
  * Number Sold

---

## ⚙️ How the App Works

1. Upload both CSV files
2. Select:

   * Training window (4–8 weeks)
   * Model (SARIMA / Gradient Boosting)
   * Zoom days
3. Click **Run Forecasts**

---

## 📊 Dashboard Sections

### Dataset Summary

* Total rows, date range, preview

### Last 4 Weeks Trends

* Recent sales patterns

### Forecast (Next 4 Weeks)

* Graph + table + zoom view

### Accuracy

* MAE, RMSE, MAPE

---

## 🧠 Forecasting Models

### SARIMA

* Seasonal time-series model
* order = (1,1,1)
* seasonal_order = (1,1,1,7)

### Gradient Boosting

* Uses lag features
* Uses rolling averages
* Captures complex patterns

---

## 📌 Important Notes

* Missing dates are filled with 0
* Forecast values cannot be negative
* Metrics require sufficient data
* Incorrect file format may cause errors

---

## 👨‍💻 Author

**Azan**
GitHub: https://github.com/azanmian676-lab

