
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MarketMind AI")
st.subheader("Explainable Financial Market Intelligence Platform")

st.markdown(
    "Financial data analysis | Machine Learning | Deep Learning | "
    "Financial NLP | Explainable AI"
)

st.info(
    "Research and educational prototype. "
    "Predictions are experimental and are not investment advice."
)

# --------------------------------------------------
# 2. DATA COLLECTION
# --------------------------------------------------

import yfinance as yf


@st.cache_data(ttl=900)
def load_market_data(ticker, period):

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


st.sidebar.header("Market Configuration")

ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="SPY"
).upper().strip()

period = st.sidebar.selectbox(
    "Historical Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

if not ticker:
    st.warning("Please enter a valid ticker.")
    st.stop()

try:
    raw = load_market_data(ticker, period)

    if raw.empty:
        st.error("No market data found. Try another ticker.")
        st.stop()

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if not all(column in raw.columns for column in required_columns):
        st.error("The downloaded dataset is missing required columns.")
        st.stop()

except Exception as error:
    st.error(f"Data collection failed: {error}")
    st.stop()


# --------------------------------------------------
# 3. DATA PREPROCESSING AND FEATURE ENGINEERING
# --------------------------------------------------

df = raw.copy()

df["daily_return"] = df["Close"].pct_change()

df["log_return"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

df["volatility_20d"] = (
    df["log_return"].rolling(20).std() * np.sqrt(252)
)

df["sma_20"] = df["Close"].rolling(20).mean()
df["sma_50"] = df["Close"].rolling(50).mean()

df["range_pct"] = (
    (df["High"] - df["Low"]) / df["Close"]
)

df["volume_change"] = df["Volume"].pct_change()

df["target_volatility"] = (
    df["log_return"].rolling(5).std().shift(-5)
    * np.sqrt(252)
)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

returns = df["log_return"].dropna()

# --------------------------------------------------
# 4. DASHBOARD METRICS
# --------------------------------------------------

latest_price = float(df["Close"].iloc[-1])

valid_volatility = df["volatility_20d"].dropna()

latest_volatility = (
    float(valid_volatility.iloc[-1])
    if not valid_volatility.empty else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Closing Price",
    f"${latest_price:,.2f}"
)

col2.metric(
    "20-Day Annualized Volatility",
    f"{latest_volatility:.2%}"
)

col3.metric(
    "Total Observations",
    f"{len(df):,}"
)

col4.metric(
    "Selected Asset",
    ticker
)

st.divider()

# --------------------------------------------------
# 5. DASHBOARD TABS
# --------------------------------------------------

tabs = st.tabs([
    "Market Overview",
    "Statistics & Regimes",
    "PCA & Linear Algebra",
    "ML Forecast",
    "PyTorch Deep Learning",
    "Financial NLP",
    "Explainable AI",
    "LangChain Research"
])

# ==================================================
# TAB 1: MARKET OVERVIEW
# ==================================================

with tabs[0]:

    st.header("Financial Market Overview")

    st.subheader("Historical Stock Prices")

    price_fig = go.Figure()

    price_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Closing Price",
            line=dict(width=2)
        )
    )

    price_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sma_20"],
            name="20-Day Moving Average"
        )
    )

    price_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sma_50"],
            name="50-Day Moving Average"
        )
    )

    price_fig.update_layout(
        template="plotly_white",
        height=450,
        xaxis_title="Date",
        yaxis_title="Adjusted Price"
    )

    st.plotly_chart(price_fig, use_container_width=True)

    st.subheader("Daily Stock Returns")

    return_fig = go.Figure()

    return_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["daily_return"],
            name="Daily Returns",
            mode="lines"
        )
    )

    return_fig.update_layout(
        template="plotly_white",
        height=350,
        xaxis_title="Date",
        yaxis_title="Daily Return"
    )

    st.plotly_chart(return_fig, use_container_width=True)

    st.subheader("Historical Volatility")

    vol_fig = go.Figure()

    vol_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["volatility_20d"],
            name="20-Day Volatility",
            fill="tozeroy"
        )
    )

    vol_fig.update_layout(
        template="plotly_white",
        height=350
    )

    st.plotly_chart(vol_fig, use_container_width=True)


# ==================================================
# TAB 2: STATISTICS AND MARKET REGIMES
# ==================================================

with tabs[1]:

    st.header("Statistical Analysis")

    stat1, stat2, stat3, stat4 = st.columns(4)

    stat1.metric(
        "Mean Return",
        f"{returns.mean():.4%}"
    )

    stat2.metric(
        "Variance",
        f"{returns.var():.6f}"
    )

    stat3.metric(
        "Standard Deviation",
        f"{returns.std():.4%}"
    )

    stat4.metric(
        "Skewness",
        f"{stats.skew(returns):.3f}"
    )

    st.subheader("Probability Distribution of Returns")

    hist_fig = go.Figure()

    hist_fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name="Return Distribution"
        )
    )

    hist_fig.update_layout(
        template="plotly_white",
        height=350
    )

    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Correlation Analysis")

    correlation_data = df[
        ["daily_return", "volatility_20d",
         "range_pct", "volume_change"]
    ].dropna()

    correlation = correlation_data.corr()

    corr_fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1
        )
    )

    corr_fig.update_layout(
        height=400,
        template="plotly_white"
    )

    st.plotly_chart(corr_fig, use_container_width=True)

    st.subheader("Hypothesis Testing")

    positive_returns = returns[returns > 0].abs()
    negative_returns = returns[returns <= 0].abs()

    if len(positive_returns) > 1 and len(negative_returns) > 1:

        test = stats.ttest_ind(
            positive_returns,
            negative_returns,
            equal_var=False
        )

        st.write("H0: Mean absolute returns are equal on positive and negative return days.")
        st.write("H1: Mean absolute returns differ between the two groups.")

        st.metric(
            "Welch t-test p-value",
            f"{test.pvalue:.6f}"
        )

        if test.pvalue < 0.05:
            st.success("Reject H0 at the 5% significance level.")
        else:
            st.info("Insufficient evidence to reject H0 at the 5% level.")

    st.caption(
        "Exploratory test only. Serial dependence and distributional "
        "assumptions should be considered before drawing conclusions."
    )

    st.divider()

    st.header("Market Regime Detection")

    regime_features = df[
        ["daily_return", "volatility_20d", "range_pct"]
    ].dropna()

    if len(regime_features) >= 30:

        scaler = StandardScaler()

        scaled_features = scaler.fit_transform(
            regime_features
        )

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(
            scaled_features
        )

        regime_data = regime_features.copy()

        regime_data["regime"] = labels.astype(str)

        regime_fig = go.Figure()

        regime_fig.add_trace(
            go.Scatter(
                x=regime_data.index,
                y=regime_data["volatility_20d"],
                mode="markers",
                marker=dict(
                    color=labels,
                    colorscale="Viridis",
                    size=7
                ),
                name="Market Regimes"
            )
        )

        regime_fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="Date",
            yaxis_title="Volatility"
        )

        st.plotly_chart(
            regime_fig,
            use_container_width=True
        )

        st.subheader("Regime Statistics")

        summary = regime_data.groupby("regime")[
            ["daily_return", "volatility_20d", "range_pct"]
        ].mean()

        for regime, row in summary.iterrows():

            st.write(
                f"**Regime {regime}** | "
                f"Average return: {row['daily_return']:.4%} | "
                f"Volatility: {row['volatility_20d']:.4f} | "
                f"Price range: {row['range_pct']:.4f}"
            )

        st.caption(
            "K-Means cluster numbers are arbitrary. They do not "
            "automatically represent bull, bear or sideways markets."
        )


# ==================================================
# TAB 3: PCA AND LINEAR ALGEBRA
# ==================================================

with tabs[2]:

    st.header("Principal Component Analysis")

    pca_features = df[
        [
            "daily_return",
            "volatility_20d",
            "range_pct",
            "volume_change"
        ]
    ].dropna()

    if len(pca_features) >= 10:

        pca_scaler = StandardScaler()

        X_scaled = pca_scaler.fit_transform(
            pca_features
        )

        pca_model = PCA(n_components=2)

        components = pca_model.fit_transform(
            X_scaled
        )

        pca_fig = go.Figure()

        pca_fig.add_trace(
            go.Scatter(
                x=components[:, 0],
                y=components[:, 1],
                mode="markers",
                marker=dict(
                    color=np.arange(len(components)),
                    colorscale="Viridis",
                    size=6
                ),
                name="Financial Observations"
            )
        )

        pca_fig.update_layout(
            template="plotly_white",
            title="Financial Data in Two Principal Components",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            height=450
        )

        st.plotly_chart(
            pca_fig,
            use_container_width=True
        )

        variance_ratio = pca_model.explained_variance_ratio_

        pc1, pc2 = st.columns(2)

        pc1.metric(
            "PC1 Explained Variance",
            f"{variance_ratio[0]:.2%}"
        )

        pc2.metric(
            "PC2 Explained Variance",
            f"{variance_ratio[1]:.2%}"
        )

        st.metric(
            "Combined Explained Variance",
            f"{sum(variance_ratio):.2%}"
        )

        st.subheader("PCA Component Loadings")

        for i, component in enumerate(
            pca_model.components_
        ):

            st.write(
                f"**PC{i + 1}:** " +
                ", ".join(
                    f"{name}: {value:.3f}"
                    for name, value in zip(
                        pca_features.columns,
                        component
                    )
                )
            )

    else:
        st.warning("Insufficient observations for PCA.")


# ==================================================
# TAB 4: MACHINE LEARNING FORECAST
# ==================================================

with tabs[3]:

    st.header("Machine Learning - Volatility Forecasting")

    st.write(
        "Random Forest Regression predicts a five-day-ahead "
        "realized-volatility proxy using historical financial features."
    )

    ml_features = [
        "daily_return",
        "volatility_20d",
        "range_pct",
        "volume_change"
    ]

    ml_data = df[
        ml_features + ["target_volatility"]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(ml_data) < 60:

        st.warning(
            "Insufficient data. Select a longer historical period."
        )

    else:

        split = int(len(ml_data) * 0.8)

        train = ml_data.iloc[:split]
        test = ml_data.iloc[split:]

        X_train = train[ml_features]
        y_train = train["target_volatility"]

        X_test = test[ml_features]
        y_test = test["target_volatility"]

        rf_model = RandomForestRegressor(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42
        )

        rf_model.fit(
            X_train,
            y_train
        )

        predictions = rf_model.predict(
            X_test
        )

        baseline = np.repeat(
            y_train.median(),
            len(y_test)
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        baseline_mae = mean_absolute_error(
            y_test,
            baseline
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Model MAE", f"{mae:.5f}")
        c2.metric("Model RMSE", f"{rmse:.5f}")
        c3.metric("Baseline MAE", f"{baseline_mae:.5f}")

        st.subheader("Actual vs Predicted Volatility")

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=y_test,
                name="Actual Volatility"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=y_test.index,
                y=predictions,
                name="Predicted Volatility"
            )
        )

        forecast_fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="Date",
            yaxis_title="Annualized Volatility"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.session_state["rf_model"] = rf_model
        st.session_state["rf_features"] = ml_features
        st.session_state["X_test"] = X_test

        st.success("Random Forest model trained successfully.")


# ==================================================
# TAB 5: PYTORCH DEEP LEARNING
# ==================================================

with tabs[4]:

    st.header("PyTorch LSTM Volatility Forecasting")

    st.write(
        "An LSTM network learns temporal patterns from "
        "historical log returns."
    )

    if st.button("Train PyTorch LSTM"):

        try:

            import torch
            import torch.nn as nn

            from torch.utils.data import TensorDataset, DataLoader

            torch.manual_seed(42)
            np.random.seed(42)

            series = df["log_return"].dropna().values.reshape(-1, 1)

            if len(series) < 100:
                st.warning("At least 100 observations are recommended.")
            else:

                scaler_lstm = MinMaxScaler()

                scaled = scaler_lstm.fit_transform(series)

                sequence_length = 20

                X_seq = []
                y_seq = []

                for i in range(
                    sequence_length,
                    len(scaled)
                ):

                    X_seq.append(
                        scaled[i-sequence_length:i]
                    )

                    y_seq.append(
                        scaled[i]
                    )

                X_seq = np.array(X_seq)
                y_seq = np.array(y_seq)

                split = int(len(X_seq) * 0.8)

                X_train = torch.tensor(
                    X_seq[:split],
                    dtype=torch.float32
                )

                y_train = torch.tensor(
                    y_seq[:split],
                    dtype=torch.float32
                )

                X_test = torch.tensor(
                    X_seq[split:],
                    dtype=torch.float32
                )

                y_test = torch.tensor(
                    y_seq[split:],
                    dtype=torch.float32
                )

                class MarketLSTM(nn.Module):

                    def __init__(self):

                        super().__init__()

                        self.lstm = nn.LSTM(
                            input_size=1,
                            hidden_size=32,
                            num_layers=1,
                            batch_first=True
                        )

                        self.dropout = nn.Dropout(0.2)

                        self.fc = nn.Linear(
                            32,
                            1
                        )

                    def forward(self, x):

                        output, _ = self.lstm(x)

                        output = output[:, -1, :]

                        output = self.dropout(output)

                        return self.fc(output)

                model = MarketLSTM()

                criterion = nn.MSELoss()

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.001
                )

                loader = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=32,
                    shuffle=False
                )

                progress = st.progress(0)

                status = st.empty()

                epochs = 15

                for epoch in range(epochs):

                    model.train()

                    total_loss = 0

                    for xb, yb in loader:

                        optimizer.zero_grad()

                        output = model(xb)

                        loss = criterion(
                            output,
                            yb
                        )

                        loss.backward()

                        optimizer.step()

                        total_loss += loss.item()

                    status.write(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Loss: {total_loss / len(loader):.6f}"
                    )

                    progress.progress(
                        (epoch + 1) / epochs
                    )

                model.eval()

                with torch.no_grad():

                    predictions = model(
                        X_test
                    ).numpy()

                actual = scaler_lstm.inverse_transform(
                    y_test.numpy()
                ).flatten()

                predicted = scaler_lstm.inverse_transform(
                    predictions
                ).flatten()

                lstm_mae = mean_absolute_error(
                    actual,
                    predicted
                )

                lstm_rmse = np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted
                    )
                )

                st.success("PyTorch LSTM training completed.")

                c1, c2 = st.columns(2)

                c1.metric(
                    "LSTM MAE",
                    f"{lstm_mae:.6f}"
                )

                c2.metric(
                    "LSTM RMSE",
                    f"{lstm_rmse:.6f}"
                )

                lstm_fig = go.Figure()

                lstm_fig.add_trace(
                    go.Scatter(
                        y=actual,
                        name="Actual Returns"
                    )
                )

                lstm_fig.add_trace(
                    go.Scatter(
                        y=predicted,
                        name="Predicted Returns"
                    )
                )

                lstm_fig.update_layout(
                    template="plotly_white",
                    height=400,
                    title="LSTM Actual vs Predicted Log Returns"
                )

                st.plotly_chart(
                    lstm_fig,
                    use_container_width=True
                )

                st.session_state["lstm_model"] = model

        except ImportError:

            st.error(
                "PyTorch is not installed. Run: "
                "pip install torch"
            )

        except Exception as error:

            st.error(
                f"LSTM training failed: {error}"
            )

    st.caption(
        "This LSTM predicts the next log return, not volatility. "
        "It is an experimental time-series model."
    )


# ==================================================
# TAB 6: FINANCIAL NLP - FINBERT
# ==================================================

with tabs[5]:

    st.header("Financial News Sentiment Analysis")

    st.write(
        "Analyze financial news using the pretrained FinBERT model."
    )

    headline = st.text_area(
        "Enter Financial News",
        value=(
            "The company reported stronger-than-expected earnings "
            "and increased its annual revenue forecast."
        )
    )

    if st.button("Analyze Financial Sentiment"):

        if not headline.strip():

            st.warning("Enter a financial headline.")

        else:

            try:

                from transformers import pipeline

                @st.cache_resource
                def load_finbert():

                    return pipeline(
                        "text-classification",
                        model="ProsusAI/finbert"
                    )

                classifier = load_finbert()

                result = classifier(
                    headline[:1500]
                )[0]

                st.success(
                    f"Sentiment: {result['label'].upper()}"
                )

                st.metric(
                    "Model Confidence",
                    f"{result['score']:.2%}"
                )

                st.caption(
                    "Model: ProsusAI/finbert"
                )

            except Exception as error:

                positive_words = [
                    "growth", "profit", "strong",
                    "beat", "surge", "raised",
                    "upgrade", "record", "gain"
                ]

                negative_words = [
                    "loss", "weak", "miss",
                    "decline", "fall", "cut",
                    "downgrade", "risk", "crisis"
                ]

                text = headline.lower()

                score = (
                    sum(word in text for word in positive_words)
                    -
                    sum(word in text for word in negative_words)
                )

                label = (
                    "positive" if score > 0
                    else "negative" if score < 0
                    else "neutral"
                )

                st.warning(
                    f"FinBERT unavailable: {error}"
                )

                st.info(
                    f"Fallback keyword sentiment: {label}"
                )

                st.caption(
                    "This fallback is a simple keyword heuristic, "
                    "not a trained financial NLP model."
                )


# ==================================================
# TAB 7: EXPLAINABLE AI - SHAP
# ==================================================

with tabs[6]:

    st.header("Explainable Artificial Intelligence")

    st.write(
        "Understand which financial features influence "
        "the Random Forest model."
    )

    if "rf_model" in st.session_state:

        rf_model = st.session_state["rf_model"]

        feature_names = st.session_state["rf_features"]

        importance = pd.Series(
            rf_model.feature_importances_,
            index=feature_names
        ).sort_values()

        importance_fig = go.Figure()

        importance_fig.add_trace(
            go.Bar(
                x=importance.values,
                y=importance.index,
                orientation="h"
            )
        )

        importance_fig.update_layout(
            template="plotly_white",
            title="Random Forest Feature Importance",
            height=350
        )

        st.plotly_chart(
            importance_fig,
            use_container_width=True
        )

        try:

            import shap

            X_explain = st.session_state["X_test"].tail(30)

            explainer = shap.TreeExplainer(
                rf_model
            )

            shap_values = explainer.shap_values(
                X_explain
            )

            shap_importance = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=feature_names
            ).sort_values()

            st.subheader("SHAP Global Feature Contributions")

            shap_fig = go.Figure()

            shap_fig.add_trace(
                go.Bar(
                    x=shap_importance.values,
                    y=shap_importance.index,
                    orientation="h"
                )
            )

            shap_fig.update_layout(
                template="plotly_white",
                height=350
            )

            st.plotly_chart(
                shap_fig,
                use_container_width=True
            )

            st.success(
                "SHAP explanations generated successfully."
            )

        except Exception as error:

            st.warning(
                f"SHAP is unavailable: {error}"
            )

            st.write(
                "The Random Forest feature importance "
                "chart above is still available."
            )

    else:

        st.info(
            "Open the ML Forecast tab first to train the model."
        )


# ==================================================
# TAB 8: LANGCHAIN MARKET RESEARCH
# ==================================================

with tabs[7]:

    st.header("AI Financial Market Research Assistant")

    st.write(
        "Prepare structured market research prompts using "
        "LangChain. An LLM provider is required for generated answers."
    )

    market_question = st.text_area(
        "Enter your market research question",
        value=(
            "Summarize the recent market behaviour, "
            "volatility and potential risk factors."
        )
    )

    context = f"""
    Asset: {ticker}
    Latest closing price: {latest_price:.2f}
    Annualized 20-day volatility: {latest_volatility:.4f}
    Mean daily log return: {returns.mean():.6f}
    Daily return standard deviation: {returns.std():.6f}
    Historical observations: {len(df)}
    """

    if st.button("Prepare LangChain Research Prompt"):

        try:

            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a financial research assistant. "
                    "Use only the supplied market context. "
                    "Clearly distinguish observations from assumptions. "
                    "Do not provide personalized investment advice."
                ),
                (
                    "human",
                    "Market context:\n{context}\n\n"
                    "Research question:\n{question}"
                )
            ])

            formatted = prompt.format_messages(
                context=context,
                question=market_question
            )

            st.success(
                "LangChain prompt template created successfully."
            )

            st.subheader("Market Context")

            st.code(
                context,
                language="text"
            )

            st.subheader("Formatted Research Prompt")

            for message in formatted:

                st.write(
                    f"**{message.type.upper()}**"
                )

                st.code(
                    message.content,
                    language="text"
                )

            st.info(
                "This creates the LangChain prompt but does not "
                "generate an LLM answer. Connect a supported chat "
                "model and API credentials to enable full generation."
            )

        except ImportError:

            st.error(
                "LangChain is not installed. Run: "
                "pip install langchain-core"
            )

        except Exception as error:

            st.error(
                f"LangChain error: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    **MarketMind AI | Financial Data Science Research Platform**

    Technologies: Python • NumPy • Pandas • SciPy •
    Scikit-learn • PyTorch • Hugging Face • FinBERT •
    LangChain • SHAP • Streamlit

    *Educational prototype. Not investment advice.*
    """
)
