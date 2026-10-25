
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    IsolationForest,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CloudBudget Guard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# DARK THEME
# =====================================================

st.markdown("""
<style>
/* Main background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #0b1120;
    color: #e5e7eb;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101a2d;
    border-right: 1px solid #263650;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0;
}
[data-testid="stSidebar"] hr {
    border-color: #293b55;
}

/* Typography */
h1, h2, h3, h4 {
    color: #f8fafc !important;
}
p, label, .stMarkdown, .stCaption {
    color: #cbd5e1;
}
[data-testid="stCaptionContainer"] {
    color: #94a3b8;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243a, #111c30);
    padding: 18px;
    border: 1px solid #2a3b55;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #a9bad1 !important;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}
[data-testid="stMetricDelta"] {
    color: #34d399 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #aebdd1;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
}
button[data-baseweb="tab"] p {
    color: inherit !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: #2563eb;
    color: #ffffff !important;
    border: 1px solid #3b82f6;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
    transition: 0.2s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1d4ed8;
    border-color: #60a5fa;
    color: #ffffff !important;
}

/* Text inputs */
[data-baseweb="input"],
[data-baseweb="textarea"] {
    background: #17243a;
    border-color: #344761;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: #f8fafc !important;
    background: transparent;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: #17243a;
    border-color: #344761;
    color: #f8fafc;
}
[data-baseweb="popover"] {
    background: #17243a;
}

/* Date picker and multiselect */
[data-testid="stDateInput"] input {
    background: #17243a;
    color: #f8fafc;
}
[data-testid="stMultiSelect"] span {
    color: #f8fafc;
}

/* Data tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #293b55;
    border-radius: 10px;
    overflow: hidden;
}

/* Alerts and information boxes */
[data-testid="stAlert"] {
    background: #142239;
    border: 1px solid #2c4566;
    border-radius: 10px;
    color: #e2e8f0;
}

/* Dividers */
hr {
    border-color: #263650;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #121e32;
    border: 1px dashed #3b5273;
    border-radius: 12px;
    padding: 12px;
}

/* Expander */
[data-testid="stExpander"] {
    background: #111c30;
    border: 1px solid #293b55;
    border-radius: 10px;
}

/* Footer */
.footer {
    color: #8494ad;
    text-align: center;
    padding: 28px 0 8px;
    font-size: 12px;
}

/* Responsive metrics */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
    background: linear-gradient(120deg, #152743, #10203a, #172b49);
    border: 1px solid #304666;
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 18px;
">
    <div style="font-size: 13px; color: #60a5fa;
                font-weight: 700; letter-spacing: 2px;">
        CLOUD FINANCIAL INTELLIGENCE
    </div>
    <h1 style="margin: 8px 0; font-size: 34px;">
        ☁️ CloudBudget Guard
    </h1>
    <p style="margin-bottom: 0; color: #b8c9df;">
        Monitor cloud spending · Forecast future costs · Detect anomalies
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Demo notice: This application uses demonstration data and may "
    "contain synthetic anomaly labels. Model results depend on the "
    "selected dataset and filters and are not a guarantee of real-world performance."
)


# =====================================================
# DATA UPLOAD
# =====================================================

DEFAULT_FILE = "cloudbudget_guard_dataset_95_demo.csv"

st.sidebar.markdown("## ☁️ CloudBudget Guard")
st.sidebar.caption("Cloud Cost Intelligence Platform")
st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "Upload cloud cost CSV",
    type=["csv"]
)

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded, low_memory=False)
    elif os.path.exists(DEFAULT_FILE):
        raw = pd.read_csv(DEFAULT_FILE, low_memory=False)
    else:
        st.warning(
            f"Dataset not found. Upload a CSV or place "
            f"{DEFAULT_FILE} beside app.py."
        )
        st.stop()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()


# =====================================================
# DATA CLEANING
# =====================================================

def normalize_data(data):
    data = data.copy()

    data.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in data.columns
    ]

    aliases = {
        "date": "charge_date",
        "usage_date": "charge_date",
        "cost_usd": "cost",
        "cloud_provider": "provider",
        "service": "service_name"
    }

    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in data.columns and new not in data.columns
    }
    data = data.rename(columns=rename_map)

    required = ["charge_date", "cost"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    data["charge_date"] = pd.to_datetime(
        data["charge_date"], errors="coerce"
    )
    data["cost"] = pd.to_numeric(data["cost"], errors="coerce")

    data = data.dropna(subset=["charge_date", "cost"])

    defaults = {
        "provider": "Unknown",
        "service_name": "Unknown",
        "team_id": "Unknown",
        "client_id": "Unknown",
        "resource_id": "Unknown",
        "currency": "USD",
        "usage_quantity": 0,
        "cpu_utilization_pct": 0,
        "memory_utilization_pct": 0,
        "network_inbound_bytes": 0,
        "network_outbound_bytes": 0,
        "anomaly_label": "normal"
    }

    for col, default in defaults.items():
        if col not in data.columns:
            data[col] = default

        if isinstance(default, str):
            data[col] = data[col].fillna(default).astype(str)
        else:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            ).fillna(default)

    data["month"] = data["charge_date"].dt.to_period("M").astype(str)
    return data


df = normalize_data(raw)

if df.empty:
    st.error("No valid records were found in the dataset.")
    st.stop()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("⚙️ Filters")

min_d = df["charge_date"].min().date()
max_d = df["charge_date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

providers = sorted(df["provider"].astype(str).unique())
services = sorted(df["service_name"].astype(str).unique())

selected_providers = st.sidebar.multiselect(
    "Cloud provider",
    providers,
    default=providers
)

selected_services = st.sidebar.multiselect(
    "Service",
    services,
    default=services
)

filtered = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[
        (filtered["charge_date"].dt.date >= date_range[0])
        & (filtered["charge_date"].dt.date <= date_range[1])
    ]

filtered = filtered[
    filtered["provider"].astype(str).isin(selected_providers)
    & filtered["service_name"].astype(str).isin(selected_services)
]

st.sidebar.divider()
st.sidebar.caption(f"Dataset records: {len(df):,}")
st.sidebar.caption(f"Filtered records: {len(filtered):,}")

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the filters.")
    st.stop()


# =====================================================
# CHART THEME
# =====================================================

DARK_TEMPLATE = "plotly_dark"

CHART_LAYOUT = {
    "paper_bgcolor": "#111c30",
    "plot_bgcolor": "#111c30",
    "font": {"color": "#e2e8f0"},
    "title_font": {"color": "#f8fafc"},
    "margin": {"l": 20, "r": 20, "t": 65, "b": 30},
    "legend": {"bgcolor": "rgba(0,0,0,0)"}
}


def style_chart(fig):
    fig.update_layout(
        template=DARK_TEMPLATE,
        **CHART_LAYOUT
    )
    fig.update_xaxes(
        gridcolor="#263650",
        linecolor="#344761",
        zerolinecolor="#344761"
    )
    fig.update_yaxes(
        gridcolor="#263650",
        linecolor="#344761",
        zerolinecolor="#344761"
    )
    return fig


# =====================================================
# NAVIGATION
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Cloud Cost Dashboard",
    "📈 Cost Forecasting",
    "🚨 Anomaly Detection"
])


# =====================================================
# FEATURE 1: CLOUD COST DASHBOARD
# =====================================================

with tab1:
    st.subheader("📊 Cloud Cost Dashboard")
    st.caption("Monitor expenditure across providers, services and teams.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Cost", f"${filtered.cost.sum():,.2f}")
    c2.metric("Average Record Cost", f"${filtered.cost.mean():,.2f}")
    c3.metric("Total Records", f"{len(filtered):,}")
    c4.metric("Services", f"{filtered.service_name.nunique():,}")

    monthly = (
        filtered.groupby("month", as_index=False)["cost"]
        .sum()
        .sort_values("month")
    )

    fig = px.line(
        monthly,
        x="month",
        y="cost",
        markers=True,
        title="Monthly Cloud Expenditure"
    )
    fig.update_traces(line=dict(color="#60a5fa", width=3))
    fig.update_layout(yaxis_title="Cost", xaxis_title="Month")
    st.plotly_chart(style_chart(fig), use_container_width=True)

    left, right = st.columns(2)

    with left:
        by_service = (
            filtered.groupby("service_name", as_index=False)["cost"]
            .sum()
            .sort_values("cost", ascending=False)
        )

        fig = px.bar(
            by_service,
            x="service_name",
            y="cost",
            title="Cost by Service",
            color="cost",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        by_provider = (
            filtered.groupby("provider", as_index=False)["cost"]
            .sum()
            .sort_values("cost", ascending=False)
        )

        fig = px.bar(
            by_provider,
            x="provider",
            y="cost",
            title="Cost by Cloud Provider",
            color="cost",
            color_continuous_scale="Teal"
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    if "team_id" in filtered.columns:
        team = (
            filtered.groupby("team_id", as_index=False)["cost"]
            .sum()
            .sort_values("cost", ascending=False)
        )

        fig = px.bar(
            team,
            x="team_id",
            y="cost",
            title="Cost by Team",
            color="cost",
            color_continuous_scale="Purples"
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.download_button(
        "⬇️ Download Filtered Data",
        filtered.to_csv(index=False).encode("utf-8"),
        "cloudbudget_filtered.csv",
        "text/csv"
    )


# =====================================================
# FEATURE 2: COST FORECASTING
# =====================================================

with tab2:
    st.subheader("📈 Cost Forecasting")
    st.caption(
        "Compare regression models and estimate future monthly cloud costs."
    )

    monthly = (
        filtered.groupby(
            pd.Grouper(key="charge_date", freq="MS")
        )["cost"]
        .sum()
        .reset_index()
        .sort_values("charge_date")
        .dropna()
    )

    if len(monthly) < 12:
        st.warning(
            "At least 12 months of data are recommended for chronological "
            "model evaluation. Select a wider date range if available."
        )
    else:
        test_n = min(6, max(3, int(round(len(monthly) * 0.2))))
        train = monthly.iloc[:-test_n].copy()
        test = monthly.iloc[-test_n:].copy()

        def make_features(series):
            return pd.DataFrame({
                "t": np.arange(len(series)),
                "month_num": series["charge_date"].dt.month.astype(int).values
            })

        X_train = make_features(train)
        X_test = pd.DataFrame({
            "t": np.arange(len(train), len(monthly)),
            "month_num": test["charge_date"].dt.month.astype(int).values
        })

        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=10,
                l2_regularization=1.0,
                random_state=42
            )
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, train["cost"].values)

            pred = np.maximum(0, model.predict(X_test))

            mae = mean_absolute_error(test["cost"], pred)
            rmse = mean_squared_error(test["cost"], pred) ** 0.5
            wape = (
                np.abs(test["cost"].values - pred).sum()
                / max(np.abs(test["cost"].values).sum(), 1e-9)
            ) * 100

            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "WAPE (%)": wape,
                "model_obj": model
            })

        results_df = pd.DataFrame([
            {k: v for k, v in result.items() if k != "model_obj"}
            for result in results
        ]).sort_values("WAPE (%)")

        st.markdown("### Model Evaluation")

        st.caption(
            "Evaluation uses the latest months as a chronological holdout. "
            "Lower WAPE, MAE and RMSE indicate lower prediction error."
        )

        st.dataframe(
            results_df.style.format({
                "MAE": "${:,.2f}",
                "RMSE": "${:,.2f}",
                "WAPE (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        best = min(results, key=lambda r: r["WAPE (%)"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", best["Model"])
        m2.metric("Holdout WAPE", f"{best['WAPE (%)']:.2f}%")
        m3.metric("Holdout MAE", f"${best['MAE']:,.2f}")

        st.info(
            "WAPE is the primary forecasting error metric. "
            "100 − WAPE may be shown as an informal accuracy-like proxy, "
            "but it is not formal classification accuracy."
        )

        # Refit the selected model on all available months.
        best_model = best["model_obj"]
        all_X = make_features(monthly)
        best_model.fit(all_X, monthly["cost"].values)

        horizons = [1, 2, 3, 6, 12]
        last_date = monthly["charge_date"].max()

        future_dates = [
            last_date + pd.offsets.MonthBegin(i)
            for i in horizons
        ]

        future_X = pd.DataFrame({
            "t": [len(monthly) - 1 + i for i in horizons],
            "month_num": [d.month for d in future_dates]
        })

        future_cost = np.maximum(0, best_model.predict(future_X))

        fc = pd.DataFrame({
            "Forecast Month": [d.strftime("%Y-%m") for d in future_dates],
            "Predicted Cost": future_cost
        })

        history_plot = monthly.rename(columns={
            "charge_date": "date",
            "cost": "value"
        }).copy()
        history_plot["Type"] = "Historical"

        forecast_plot = pd.DataFrame({
            "date": future_dates,
            "value": future_cost,
            "Type": "Forecast"
        })

        combined = pd.concat(
            [history_plot, forecast_plot],
            ignore_index=True
        )

        fig = px.line(
            combined,
            x="date",
            y="value",
            color="Type",
            markers=True,
            title="Historical Spending and Forecast"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Predicted Cost"
        )

        st.plotly_chart(style_chart(fig), use_container_width=True)

        st.markdown("### Future Cost Estimates")

        st.dataframe(
            fc.style.format({"Predicted Cost": "${:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Forecast",
            fc.to_csv(index=False).encode("utf-8"),
            "cloudbudget_forecast.csv",
            "text/csv"
        )


# =====================================================
# FEATURE 3: ANOMALY DETECTION
# =====================================================

with tab3:
    st.subheader("🚨 Anomaly Detection")
    st.caption(
        "Identify unusual expenditure patterns and evaluate predictions "
        "against available anomaly labels."
    )

    label_col = (
        "anomaly_label"
        if "anomaly_label" in filtered.columns
        else None
    )

    if label_col and filtered[label_col].nunique() >= 2:
        model_df = filtered.copy()

        y = (
            model_df[label_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .ne("normal")
            .astype(int)
        )

        numeric_features = [
            col for col in [
                "cost",
                "usage_quantity",
                "cpu_utilization_pct",
                "memory_utilization_pct",
                "network_inbound_bytes",
                "network_outbound_bytes"
            ]
            if col in model_df.columns
        ]

        categorical_features = [
            col for col in [
                "provider",
                "service_name",
                "team_id",
                "region"
            ]
            if col in model_df.columns
        ]

        X = model_df[numeric_features + categorical_features].copy()

        for col in numeric_features:
            X[col] = (
                pd.to_numeric(X[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

        for col in categorical_features:
            X[col] = X[col].fillna("Unknown").astype(str)

        if y.nunique() < 2 or y.value_counts().min() < 2:
            st.warning(
                "There are not enough examples of both classes in the "
                "selected data. Expand the filters to evaluate the model."
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            transformers = []

            if numeric_features:
                transformers.append((
                    "num",
                    "passthrough",
                    numeric_features
                ))

            if categorical_features:
                transformers.append((
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                ))

            pre = ColumnTransformer(transformers)

            clf = Pipeline([
                ("prep", pre),
                ("model", ExtraTreesClassifier(
                    n_estimators=250,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)

            st.markdown("### Model Performance")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Holdout Accuracy", f"{acc * 100:.2f}%")
            m2.metric("Anomaly Precision", f"{prec * 100:.2f}%")
            m3.metric("Anomaly Recall", f"{rec * 100:.2f}%")
            m4.metric("Anomaly F1-score", f"{f1 * 100:.2f}%")

            st.caption(
                "Metrics are calculated on the test split of the selected "
                "data. They may change when filters or data change."
            )

            st.markdown("### Confusion Matrix")

            cm = confusion_matrix(y_test, pred, labels=[0, 1])

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Normal", "Actual Anomaly"],
                columns=["Predicted Normal", "Predicted Anomaly"]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            # Score all currently selected records using the fitted model.
            scored = model_df.copy()
            scored["predicted_anomaly"] = clf.predict(X)
            detected = (
                scored[scored["predicted_anomaly"] == 1]
                .sort_values("cost", ascending=False)
            )

            st.markdown("### Detected Anomalies")

            st.metric("Potential Anomalies Detected", f"{len(detected):,}")

            st.dataframe(
                detected.head(200),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Detected Anomalies",
                detected.to_csv(index=False).encode("utf-8"),
                "detected_anomalies.csv",
                "text/csv"
            )

    else:
        st.warning(
            "This dataset does not contain usable anomaly labels. "
            "Potential anomalies can be identified, but verified "
            "classification accuracy cannot be calculated."
        )

        feature_cols = [
            "cost",
            "usage_quantity",
            "cpu_utilization_pct",
            "memory_utilization_pct",
            "network_inbound_bytes",
            "network_outbound_bytes"
        ]

        features = filtered[feature_cols].copy()

        for col in features.columns:
            features[col] = (
                pd.to_numeric(features[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

        if len(features) >= 10 and features.nunique().max() > 1:
            iso = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=42
            )

            out = iso.fit_predict(features)

            shown = filtered.copy()
            shown["predicted_anomaly"] = out == -1

            st.metric(
                "Potential Anomalies Found",
                f"{int((out == -1).sum()):,}"
            )

            st.dataframe(
                shown[shown["predicted_anomaly"]].head(200),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Potential Anomalies",
                shown[shown["predicted_anomaly"]]
                .to_csv(index=False).encode("utf-8"),
                "potential_anomalies.csv",
                "text/csv"
            )
        else:
            st.warning(
                "Not enough varied data is available for anomaly detection."
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    CloudBudget Guard · Cloud Cost Monitoring and Analytics<br>
    Forecasts and anomaly flags are decision-support outputs and should
    be validated before operational use.
</div>
""", unsafe_allow_html=True)
