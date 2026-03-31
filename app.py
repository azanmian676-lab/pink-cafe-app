import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================================================
# 1) READ CSV FILES
# =========================================================

def read_coffee_csv(path: str) -> pd.DataFrame:
    """
    Reads: Pink CoffeeSales March - Oct 2025.csv
    """
    raw = pd.read_csv(path, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]

    # First row contains coffee names
    # raw.iloc[0] -> Date=NaN, Number Sold="Cappuccino", Unnamed:2="Americano"
    coffee_names = []
    coffee_cols = []

    # We take all columns except Date as possible coffee columns
    for col in raw.columns:
        if col == "Date":
            continue
        name = str(raw.loc[0, col]).strip()
        if name.lower() in ("nan", "", "none"):
            continue
        coffee_names.append(name)
        coffee_cols.append(col)

    # Data starts from row 1 onward
    data = raw.iloc[1:].copy()

    # Parse dates (dayfirst)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")

    # Keep only Date + coffee columns found
    keep = ["Date"] + coffee_cols
    data = data[keep].copy()

    # Melt to long
    long_df = data.melt(id_vars=["Date"], value_vars=coffee_cols, var_name="col", value_name="qty")
    long_df["item"] = long_df["col"].map(dict(zip(coffee_cols, coffee_names)))
    long_df = long_df.drop(columns=["col"])
    long_df = long_df.rename(columns={"Date": "date"})

    # Clean qty
    long_df["qty"] = pd.to_numeric(long_df["qty"], errors="coerce")
    long_df = long_df.dropna(subset=["qty"])
    long_df["qty"] = long_df["qty"].astype(float)

    long_df = long_df.sort_values(["item", "date"]).reset_index(drop=True)
    return long_df[["date", "item", "qty"]]


def read_croissant_csv(path: str) -> pd.DataFrame:
    """
    Reads: Pink CroissantSales March - Oct 2025.csv
    """
    raw = pd.read_csv(path, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]

    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["Date"]).sort_values("Date")

    raw["Number Sold"] = pd.to_numeric(raw["Number Sold"], errors="coerce")
    raw = raw.dropna(subset=["Number Sold"])

    out = raw.rename(columns={"Date": "date", "Number Sold": "qty"}).copy()
    out["item"] = "Croissant"
    out["qty"] = out["qty"].astype(float)

    return out[["date", "item", "qty"]]


def load_all_data(coffee_path: str, croissant_path: str) -> pd.DataFrame:
    coffee = read_coffee_csv(coffee_path)
    food = read_croissant_csv(croissant_path)
    df = pd.concat([coffee, food], ignore_index=True)
    return df.sort_values(["item", "date"]).reset_index(drop=True)


# =========================================================
# 2) HELPERS: FILTERING + TOP ITEMS
# =========================================================

def filter_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["date"].max()
    start = end - pd.Timedelta(days=days)
    return df[(df["date"] > start) & (df["date"] <= end)].copy()


def top_items(df: pd.DataFrame, items_list: list, n: int = 3) -> list:
    """
    Returns top n items among a provided list, based on total qty.
    """
    subset = df[df["item"].isin(items_list)]
    totals = subset.groupby("item")["qty"].sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


# =========================================================
# 3) FORECASTING MODELS (SARIMA + ML)
# =========================================================

def ensure_daily_series(df_item: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure continuous daily series.
    Missing days -> qty=0
    """
    s = df_item[["date", "qty"]].copy().sort_values("date").set_index("date")
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(idx)
    s["qty"] = s["qty"].fillna(0.0)
    return s.reset_index().rename(columns={"index": "date"})


def time_split(s: pd.DataFrame, test_days: int = 14):
    if len(s) <= test_days + 7:
        return s.copy(), s.iloc[0:0].copy()
    split = len(s) - test_days
    return s.iloc[:split].copy(), s.iloc[split:].copy()


def calc_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def forecast_sarima(df_item: pd.DataFrame, horizon_days: int = 28):
    """
    Weekly seasonality SARIMA.
    """
    s = ensure_daily_series(df_item)
    train, test = time_split(s, test_days=14)

    y_train = train["qty"].values

    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=50)

    metrics = {}
    if not test.empty:
        pred_test = fit.forecast(steps=len(test))
        metrics = calc_metrics(test["qty"].values, pred_test)

    # Fit on full for final forecast
    y_full = s["qty"].values
    model_full = SARIMAX(
        y_full,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_full = model_full.fit(disp=False, maxiter=50)

    forecast_vals = fit_full.forecast(steps=horizon_days)
    forecast_vals = np.clip(forecast_vals, 0, None)

    last_date = s["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    fc = pd.DataFrame({"date": future_dates, "yhat": forecast_vals})
    return s, fc, "SARIMA", metrics


def make_lag_features(s: pd.DataFrame, max_lag: int = 14):
    df = s.copy().sort_values("date").set_index("date")

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["qty"].shift(lag)

    df["roll_mean_7"] = df["qty"].shift(1).rolling(7).mean()
    df["dow"] = df.index.dayofweek

    df = df.dropna().reset_index()
    return df


def forecast_gb(df_item: pd.DataFrame, horizon_days: int = 28, max_lag: int = 14):
    """
    GradientBoosting with lag features, rolling mean, and day-of-week.
    """
    s = ensure_daily_series(df_item)
    feats = make_lag_features(s, max_lag=max_lag)

    # Fallback if too little data
    if len(feats) < 30:
        last_date = s["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        mean7 = float(s["qty"].tail(7).mean()) if len(s) >= 7 else float(s["qty"].mean())
        fc = pd.DataFrame({"date": future_dates, "yhat": np.full(horizon_days, max(0.0, mean7))})
        return s, fc, "GB_FallbackMean7", {}

    # Split
    feats = feats.sort_values("date")
    split = max(1, len(feats) - 14)
    train_df = feats.iloc[:split]
    test_df = feats.iloc[split:]

    feature_cols = [c for c in feats.columns if c.startswith("lag_")] + ["roll_mean_7", "dow"]

    X_train, y_train = train_df[feature_cols].values, train_df["qty"].values
    X_test, y_test = test_df[feature_cols].values, test_df["qty"].values

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    metrics = {}
    if len(test_df) >= 5:
        pred_test = model.predict(X_test)
        metrics = calc_metrics(y_test, pred_test)

    # Fit on full
    model.fit(feats[feature_cols].values, feats["qty"].values)

    # Iterative future prediction
    history = s[["date", "qty"]].copy().sort_values("date").reset_index(drop=True)
    preds = []

    for _ in range(horizon_days):
        last_date = history["date"].max()
        next_date = last_date + pd.Timedelta(days=1)

        qty_series = history.set_index("date")["qty"]

        row = {"dow": next_date.dayofweek}
        for lag in range(1, max_lag + 1):
            row[f"lag_{lag}"] = float(qty_series.iloc[-lag]) if len(qty_series) >= lag else float(qty_series.mean())
        row["roll_mean_7"] = float(qty_series.tail(7).mean()) if len(qty_series) >= 7 else float(qty_series.mean())

        X_next = np.array([[row[c] for c in feature_cols]])
        y_next = float(model.predict(X_next)[0])
        y_next = max(0.0, y_next)

        preds.append({"date": next_date, "yhat": y_next})

        history = pd.concat([history, pd.DataFrame({"date": [next_date], "qty": [y_next]})], ignore_index=True)

    fc = pd.DataFrame(preds)
    return s, fc, "GradientBoosting", metrics


# =========================================================
# 4) STREAMLIT DASHBOARD UI
# =========================================================

st.set_page_config(page_title="Bristol-Pink Dashboard", layout="wide")
st.title("🥐☕ Bristol-Pink Bakery Sales Prediction Dashboard")

st.write(
    "This dashboard reads two CSV files (Coffee + Croissant), shows the last 4 weeks of sales, "
    "and forecasts the next 4 weeks. It also evaluates algorithm accuracy (holdout metrics)."
)

with st.sidebar:
    st.header("Files (dataset)")
    coffee_file = st.file_uploader("Upload Pink CoffeeSales CSV", type=["csv"])
    croissant_file = st.file_uploader("Upload Pink CroissantSales CSV", type=["csv"])

    st.header("Forecast Settings")
    training_weeks = st.slider("Training window (weeks)", 4, 8, 8, 1)
    algorithms = st.multiselect("Algorithms", ["SARIMA", "GradientBoosting"], default=["SARIMA", "GradientBoosting"])
    zoom_days = st.slider("Zoom (days of forecast)", 3, 28, 10, 1)

    run_btn = st.button("Run Forecasts")


# Must have both files
if coffee_file is None or croissant_file is None:
    st.warning("Please upload BOTH CSV files (CoffeeSales and CroissantSales) in the sidebar.")
    st.stop()

# Read uploaded content by saving into pandas directly
coffee_bytes = coffee_file.getvalue()
cro_bytes = croissant_file.getvalue()

# Use pandas read_csv on bytes via BytesIO
from io import BytesIO
coffee_path_like = BytesIO(coffee_bytes)
cro_path_like = BytesIO(cro_bytes)

df = pd.concat([read_coffee_csv(coffee_path_like), read_croissant_csv(cro_path_like)], ignore_index=True)
df = df.sort_values(["item", "date"]).reset_index(drop=True)

st.subheader("Dataset Summary")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total rows", len(df))
with c2:
    st.metric("Start date", df["date"].min().date().isoformat())
with c3:
    st.metric("End date", df["date"].max().date().isoformat())

st.dataframe(df.head(20), use_container_width=True)

# Items available
coffee_items = [x for x in df["item"].unique() if x in ["Cappuccino", "Americano"]]
food_items = [x for x in df["item"].unique() if x == "Croissant"]

top_coffees = top_items(df, coffee_items, n=3)
top_foods = top_items(df, food_items, n=3)

st.write("**Top coffees (max 3):**", ", ".join(top_coffees))
st.write("**Top foods (max 3):**", ", ".join(top_foods))

selected_items = top_coffees + top_foods


tab1, tab2, tab3 = st.tabs(["📈 Last 4 Weeks Trends", "🔮 Next 4 Weeks Forecast", "✅ Accuracy"])


# ---------------------------------------------------------
# TAB 1: last 4 weeks fluctuations
# ---------------------------------------------------------
with tab1:
    st.header("Sales Fluctuation — Last 4 Weeks")
    last4 = filter_last_days(df, days=28)

    plot_df = last4[last4["item"].isin(selected_items)].copy()
    if plot_df.empty:
        st.warning("No data available for last 4 weeks.")
    else:
        fig = px.line(plot_df, x="date", y="qty", color="item", markers=True)
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plot_df.sort_values(["item", "date"]), use_container_width=True)


# ---------------------------------------------------------
# Forecast runner
# ---------------------------------------------------------
def run_all_forecasts(df, items, training_weeks, algorithms):
    # filter training window
    train_days = training_weeks * 7
    train_df = filter_last_days(df, days=train_days)

    results = {}
    for item in items:
        item_df = train_df[train_df["item"] == item][["date", "qty"]].copy()
        if item_df.empty:
            continue

        results[item] = {}

        if "SARIMA" in algorithms:
            hist, fc, name, metrics = forecast_sarima(item_df, horizon_days=28)
            results[item][name] = {"hist": hist, "fc": fc, "metrics": metrics}

        if "GradientBoosting" in algorithms:
            hist, fc, name, metrics = forecast_gb(item_df, horizon_days=28)
            results[item][name] = {"hist": hist, "fc": fc, "metrics": metrics}

    return results


# ---------------------------------------------------------
# TAB 2: forecasts (only after button click)
# ---------------------------------------------------------
with tab2:
    st.header("Predicted Sales — Next 4 Weeks")

    if not run_btn:
        st.info("Click **Run Forecasts** in the sidebar to generate predictions.")
    else:
        with st.spinner("Training models and generating forecasts..."):
            results = run_all_forecasts(df, selected_items, training_weeks, algorithms)

        if not results:
            st.warning("No forecast results produced.")
        else:
            item_pick = st.selectbox("Select item", options=list(results.keys()))
            model_pick = st.selectbox("Select model", options=list(results[item_pick].keys()))

            hist = results[item_pick][model_pick]["hist"].copy()
            fc = results[item_pick][model_pick]["fc"].copy()

            hist_plot = hist.copy()
            hist_plot["series"] = "history"
            fc_plot = fc.rename(columns={"yhat": "qty"}).copy()
            fc_plot["series"] = "forecast"

            combined = pd.concat([hist_plot, fc_plot], ignore_index=True)

            fig = px.line(combined, x="date", y="qty", color="series", markers=True)
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Forecast Table (28 days)")
            st.dataframe(fc, use_container_width=True)

            st.subheader(f"Zoom Forecast (next {zoom_days} days)")
            zoom = fc.head(zoom_days)
            fig2 = px.line(zoom, x="date", y="yhat", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(zoom, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: accuracy
# ---------------------------------------------------------
with tab3:
    st.header("Model Accuracy (Holdout Metrics)")

    if not run_btn:
        st.info("Click **Run Forecasts** to compute accuracy metrics.")
    else:
        # Use cached results by rerunning quickly
        results = run_all_forecasts(df, selected_items, training_weeks, algorithms)

        rows = []
        for item, models in results.items():
            for model_name, payload in models.items():
                m = payload["metrics"]
                rows.append({
                    "item": item,
                    "model": model_name,
                    "MAE": m.get("MAE", None),
                    "RMSE": m.get("RMSE", None),
                    "MAPE_%": m.get("MAPE_%", None),
                })

        acc = pd.DataFrame(rows)
        if acc.empty:
            st.warning("Not enough data to compute holdout metrics.")
        else:
            st.dataframe(acc.sort_values(["item", "MAPE_%"]), use_container_width=True)
            fig = px.bar(acc.dropna(), x="item", y="MAPE_%", color="model", barmode="group")
            st.plotly_chart(fig, use_container_width=True)