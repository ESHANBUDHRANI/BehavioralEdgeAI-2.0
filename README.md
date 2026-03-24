# 🧠 Explainable Risk Attribution with Agentic System
**Behavioral Trading Analysis Pipeline**

An end-to-end analytical engine designed to ingest raw trade history, enrich it with technical market context, and apply a massive suite of machine learning models to extract deep behavioral insights. The system culminates in an interactive, Agentic Retrieval-Augmented Generation (RAG) chatbot powered by Groq, allowing users to converse directly with their trading data.

---

## 🚀 Core Pipeline Overview

1. **Data Ingestion & Normalization:** Upload raw CSV trade data. The system auto-detects columns, applies FIFO (First-In-First-Out) position reconstruction, and filters out flagged "emergency" trades to ensure clean behavioral modeling.
2. **Market Context Enrichment:** Fetches historical OHLCV data via `yfinance` and computes 15+ technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.) using `ta-lib` to classify the precise market regime at the time of each trade.
3. **Behavioral Feature Engineering:** Derives psychological metrics such as *Revenge Score*, *Emotional State* (Calm, Anxious, Euphoric), and *Position Size Deviation*.
4. **Machine Learning Suite:** Runs 12 distinct mathematical and ML models to profile risk, detect anomalies, and establish causality.
5. **Interactive RAG Chatbot:** Embeds the analysis into ChromaDB and utilizes a Groq-powered LLM with specialized "Agents" to answer complex queries and run counterfactual scenarios.

---

## 🔬 Machine Learning & Mathematical Concepts

This project leverages a diverse stack of predictive and descriptive models to break down trader psychology and risk.

### 1. Clustering & State Modeling
* **Gaussian Mixture Models (GMM) & Agglomerative Clustering:** Segments trades into behavioral archetypes (e.g., *Reactive Emotional*, *Balanced Tactical*) by grouping trades with similar emotional scores, frequencies, and size deviations.
* **Hidden Markov Models (HMM):** Models the trader's underlying psychological state over time, identifying latent behavioral regimes and the transition probabilities between them.

### 2. Anomaly & Pattern Detection
* **Dual Anomaly Detection (Isolation Forest + Deep Autoencoder):** Combines traditional tree-based isolation with neural network reconstruction error to flag highly unusual trades that deviate from the user's baseline behavior.
* **LSTM Sequential Neural Network:** Analyzes the sequence of the last 5 trades to predict the Profit/Loss (PnL) outcome of the next trade based on temporal behavioral patterns.

### 3. Risk & Behavioral Finance Theory
* **Prospect Theory Curve Fitting:** Calculates the user's specific risk-seeking Alpha (α) and Loss Aversion Lambda (λ) by fitting their return distribution to Kahneman and Tversky's foundational behavioral economics model.
* **Disposition Effect:** Quantifies the tendency to hold losing positions too long and sell winning positions too early (PGR/PLR ratio).
* **GARCH Volatility Modeling:** Models conditional volatility (time-varying risk) of the trader's PnL, identifying periods of clustered behavioral risk.
* **Gaussian Copulas:** Models the complex dependency structures between position sizing and market volatility.

### 4. Causal Inference & Explainability
* **Bayesian Networks:** Constructs a directed acyclic graph to determine the conditional probabilities between emotional states, trading frequency spikes, and trade outcomes (win/loss).
* **Granger Causality:** Conducts statistical hypothesis testing to determine if specific market conditions (e.g., ATR volatility) temporally "cause" behavioral shifts (e.g., position size deviations).
* **SHAP (SHapley Additive exPlanations):** Uses a Random Forest explainer to determine the top feature contributions driving the trader's dominant behavioral cluster, offering pure explainable AI (XAI).

---

## 🤖 Agentic RAG Architecture

The final stage of the pipeline transforms the complex mathematical outputs into a natural language interface.

* **Vector Database (ChromaDB):** Stores two distinct collections:
  * *Session Data:* The exact metrics, cluster results, and PnL data from the current analysis.
  * *Static Knowledge:* A built-in repository of behavioral finance theory (e.g., definitions of the Kelly Criterion, Anchoring Bias, etc.).
* **Specialized Agents:** Before querying the LLM, the system classifies the user's intent and routes it to specialized Python functions (Behavior Agent, Risk Agent, Counterfactual Agent) to inject highly structured context.
* **Groq LLM (`llama-3.3-70b-versatile`):** Generates lightning-fast, highly contextualized advice and explanations based on the mathematical profile of the trader.

---

## 📈 Visualizations
Generates 9 interactive Plotly dashboards, including:
1. PnL & Emotional State Over Time
2. HMM Behavioral State Sequence
3. Behavioral Cluster Map
4. Return Distribution (with VaR95 and CVaR95)
5. Behavioral Bias Radar Chart
6. Anomaly Trade Detection
7. Win Rate by Market Regime
8. Prospect Theory Value Function Curve
9. PCA Cluster Projection (2D)

---

## 🛠️ Tech Stack
* **Core:** Python, Pandas, NumPy, SQLite
* **Machine Learning:** Scikit-Learn, PyTorch, HMMlearn, Pgmpy (Bayesian), Arch (GARCH), Copulas, Ruptures
* **Time-Series & Causal:** Statsmodels
* **Market Data:** yfinance, TA-Lib
* **Explainability:** SHAP
* **LLM & AI:** LangChain, Groq API, ChromaDB, Sentence-Transformers
* **Visualization:** Plotly, IPyWidgets

---

## ⚙️ How to Run
1. Open the notebook in **Google Colab**.
2. Go to `Runtime > Secrets` and add your Groq API key under the variable `GROQ_API_KEY`.
3. Run the cells sequentially from top to bottom.
4. When prompted, upload your trading history CSV.
5. Interact with the Plotly charts and the Groq-powered Chatbot at the end of the notebook!
