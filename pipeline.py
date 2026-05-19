"""
BehavioralEdge — Analysis Pipeline & API Server
Run CLI: python pipeline.py --csv your_trades.csv
Run API: python pipeline.py --serve
"""

import os, sys, json, sqlite3, uuid, warnings, argparse, tempfile
from collections import deque
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, skew, kurtosis
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hmmlearn import hmm
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import yfinance as yf
import ta as ta_lib
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# FastAPI Imports for Real-Time Integration
try:
    from fastapi import FastAPI, UploadFile, File, BackgroundTasks
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ─────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────
BASE_PATH = Path("behavioral_analysis")
BASE_PATH.mkdir(exist_ok=True)
(BASE_PATH / "chroma_db").mkdir(exist_ok=True)
(BASE_PATH / "cache").mkdir(exist_ok=True)

DB_PATH = BASE_PATH / "trades.db"
device  = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────
# SQLITE SCHEMA
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, timestamp TEXT, symbol TEXT,
        side TEXT, quantity REAL, price REAL,
        pnl REAL, holding_duration REAL, emergency INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS market_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, symbol TEXT, date TEXT,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        sma20 REAL, sma50 REAL, ema20 REAL, ema50 REAL,
        rsi REAL, macd REAL, macd_signal REAL, atr REAL,
        bb_upper REAL, bb_lower REAL, adx REAL,
        obv REAL, volume_z REAL, sentiment_score REAL,
        sentiment_label TEXT, market_regime TEXT
    );
    """)
    conn.commit()
    return conn

# ─────────────────────────────────────────
# DATA INGESTION & RECONSTRUCTION
# ─────────────────────────────────────────
AUTO_MAP = {
    'timestamp': ['timestamp','date','datetime','time','trade_date','date/time'],
    'symbol':    ['symbol','ticker','instrument','stock','scrip'],
    'side':      ['side','buy_sell','type','direction','action','buy/sell'],
    'quantity':  ['quantity','qty','shares','units','volume','lot'],
    'price':     ['price','avg_price','fill_price','trade_price','rate'],
}

def load_csv(path):
    df_raw = pd.read_csv(path)
    mapping = {}
    for canonical, aliases in AUTO_MAP.items():
        for col in df_raw.columns:
            if col.strip().lower().replace(' ', '_') in aliases:
                mapping[col] = canonical
                break
    df = df_raw.rename(columns=mapping).copy()
    required = ['timestamp','symbol','side','quantity','price']
    missing  = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['symbol']    = df['symbol'].astype(str).str.upper().str.strip()
    df['side']      = df['side'].astype(str).str.upper().str.strip()
    df['quantity']  = pd.to_numeric(df['quantity'], errors='coerce')
    df['price']     = pd.to_numeric(df['price'],    errors='coerce')
    df = df.dropna(subset=required).drop_duplicates().sort_values('timestamp').reset_index(drop=True)
    return df

def reconstruct_trades(df, sid, conn):
    sides = df['side'].unique()
    if 'BUY' in sides and 'SELL' in sides:
        return _fifo(df, sid, conn)
    elif 'pnl' in df.columns:
        realized = df[['timestamp','symbol','side','quantity','price','pnl']].copy()
        realized['session_id']       = sid
        realized['holding_duration'] = df.get('holding_duration', pd.Series(np.zeros(len(df))))
        realized['emergency']        = 0
        realized['timestamp']        = realized['timestamp'].astype(str)
        realized.to_sql('trades', conn, if_exists='append', index=False)
        conn.commit()
        return realized
    else:
        raise ValueError("CSV must have BUY+SELL rows or a 'pnl' column.")

def _fifo(df, sid, conn):
    inventory, realized = {}, []
    for _, row in df.sort_values('timestamp').iterrows():
        sym, side, qty, price = row['symbol'], row['side'].upper(), row['quantity'], row['price']
        if sym not in inventory: inventory[sym] = {'net_qty': 0, 'lots': deque()}
        inv = inventory[sym]
        is_opening = not inv['lots'] or ((inv['net_qty'] > 0 and side == 'BUY') or (inv['net_qty'] < 0 and side == 'SELL'))
        if is_opening:
            inv['lots'].append({'price': price, 'qty': qty, 'time': row['timestamp'], 'side': side})
            inv['net_qty'] += qty if side == 'BUY' else -qty
        else:
            sell_qty = qty
            wpnl, whold, matched = 0.0, 0.0, 0.0
            while sell_qty > 1e-6 and inv['lots']:
                lot = inv['lots'][0]
                match_qty = min(lot['qty'], sell_qty)
                lot['qty'] -= match_qty; sell_qty -= match_qty; matched += match_qty
                wpnl += (price - lot['price']) * match_qty if lot['side'] == 'BUY' else (lot['price'] - price) * match_qty
                whold += (row['timestamp'] - lot['time']).total_seconds() / 86400 * match_qty
                if lot['qty'] <= 1e-6: inv['lots'].popleft()
            inv['net_qty'] += qty if side == 'BUY' else -qty
            if matched > 0:
                realized.append({
                    'session_id': sid, 'timestamp': str(row['timestamp']), 'symbol': sym, 'side': side,
                    'quantity': matched, 'price': price, 'pnl': wpnl,
                    'holding_duration': whold / matched, 'emergency': 0
                })
    if realized:
        pd.DataFrame(realized).to_sql('trades', conn, if_exists='append', index=False)
        conn.commit()
    return pd.DataFrame(realized)

def _yf_fetch(sym, exchange_map, start, end):
    """Try NSE then BSE. Returns (DataFrame, ticker_used) or (empty_df, sym)."""
    exch   = exchange_map.get(sym, 'NSE')
    suffix = '.BO' if exch == 'BSE' else '.NS'
    for tick in [sym + suffix, sym + ('.NS' if suffix == '.BO' else '.BO')]:
        try:
            raw = yf.download(tick, start=start, end=end,
                              interval='1d', progress=False, auto_adjust=True)
            if not raw.empty:
                return raw, tick
        except Exception:
            pass
    return pd.DataFrame(), sym

def _estimate_pnl_from_market(df, session_id, conn):
    """
    For each SELL row, fetch the stock's closing price ~30 days before
    the sell date as a proxy buy price.
    P&L = (sell_price - est_buy_price) * quantity
    Falls back to 0 if yfinance can't find data.
    """
    exchange_map = {}
    if 'exchange' in df.columns:
        for _, row in df[['symbol', 'exchange']].drop_duplicates().iterrows():
            exchange_map[row['symbol']] = row['exchange']

    sells   = df[df['side'] == 'SELL'].copy()
    records = []

    for _, row in sells.iterrows():
        sym     = row['symbol']
        sell_ts = pd.Timestamp(row['timestamp'])
        start   = (sell_ts - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
        end     = (sell_ts - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        est_buy_price = row['price']  # break-even fallback
        try:
            raw, _ = _yf_fetch(sym, exchange_map, start, end)
            if not raw.empty:
                cols = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                raw.columns = [str(c).strip().title() for c in cols]
                close_col = 'Close' if 'Close' in raw.columns else raw.columns[0]
                est_buy_price = float(raw[close_col].iloc[-1])
        except Exception:
            pass

        pnl = (row['price'] - est_buy_price) * row['quantity']
        records.append({
            'session_id':       session_id,
            'timestamp':        str(row['timestamp']),
            'symbol':           sym,
            'side':             'SELL',
            'quantity':         row['quantity'],
            'price':            row['price'],
            'pnl':              pnl,
            'holding_duration': 30.0,
            'emergency':        0,
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result.to_sql('trades', conn, if_exists='append', index=False)
        conn.commit()
    print(f"      {len(result)} trades with estimated P&L (proxy buy = 30-day prior close)")
    return result

# ─────────────────────────────────────────
# MARKET DATA & ENRICHMENT
# ─────────────────────────────────────────
def compute_indicators(df):
    if df.empty: return df
    df.columns = [str(c[0]).strip().title() if isinstance(c, tuple) else str(c).strip().title() for c in df.columns]
    if 'Date' not in df.columns: df = df.reset_index()
    c, h, l, v = 'Close', 'High', 'Low', 'Volume'
    df['SMA50']            = ta_lib.trend.sma_indicator(df[c], 50)
    df['EMA20']            = ta_lib.trend.ema_indicator(df[c], 20)
    df['ADX_14']           = ta_lib.trend.ADXIndicator(df[h], df[l], df[c]).adx()
    df['RSI']              = ta_lib.momentum.rsi(df[c], 14)
    df['ATR']              = ta_lib.volatility.average_true_range(df[h], df[l], df[c])
    df['ATR_Rolling_Mean'] = df['ATR'].rolling(20).mean()
    return df.replace([np.inf, -np.inf], np.nan)

def fetch_market_data(modeling_df):
    exchange_map = {}
    if 'exchange' in modeling_df.columns:
        for _, row in modeling_df[['symbol', 'exchange']].drop_duplicates().iterrows():
            exchange_map[row['symbol']] = row['exchange']

    symbols = modeling_df['symbol'].unique().tolist()
    start   = (pd.to_datetime(modeling_df['timestamp']).min() - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
    end     = (pd.to_datetime(modeling_df['timestamp']).max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    market_data = {}

    for sym in symbols:
        raw, yf_sym = _yf_fetch(sym, exchange_map, start, end)
        if raw.empty:
            print(f"⚠️  {sym}: no market data on NSE or BSE — skipping enrichment.")
            continue
        try:
            raw = compute_indicators(raw)
            raw['market_regime'] = np.where(
                raw['RSI'] < 35, 'risk_off',
                np.where(raw['RSI'] > 65, 'trending_bullish', 'normal')
            )
            raw.index = pd.to_datetime(
                raw['Date'] if 'Date' in raw.columns else raw.index, errors='coerce'
            )
            market_data[sym] = raw
            print(f"✅ {sym} ({yf_sym}): {len(raw)} days fetched")
        except Exception as e:
            print(f"⚠️  {sym}: indicator computation failed — {e}")

    return market_data

def enrich_trades(modeling_df, market_data, session_id, conn):
    enriched_list = []
    for _, trade in modeling_df.iterrows():
        sym = trade['symbol']
        if sym not in market_data:
            enriched_list.append(trade.to_dict())
            continue
        mdf = market_data[sym].copy()
        if mdf.index.tz is not None:
            mdf.index = mdf.index.tz_localize(None)
        ts        = pd.Timestamp(trade['timestamp']).tz_localize(None)
        available = mdf[mdf.index <= ts]
        if available.empty:
            enriched_list.append(trade.to_dict())
            continue
        enriched_list.append({**trade.to_dict(), **available.iloc[-1].to_dict()})
    return pd.DataFrame(enriched_list)

def compute_all_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['position_value'] = df['quantity'] * df['price']
    pv_mean = df['position_value'].rolling(20, min_periods=1).mean()
    pv_std  = df['position_value'].rolling(20, min_periods=1).std().replace(0, np.nan)
    df['size_deviation']  = (df['position_value'] - pv_mean) / pv_std
    df['is_loss']         = (df['pnl'] < 0).astype(int)

    df['last_loss_time']  = df['timestamp'].where(df['is_loss'] == 1).ffill().shift(1)
    df['post_loss_hours'] = (
        pd.to_datetime(df['timestamp']) - pd.to_datetime(df['last_loss_time'])
    ).dt.total_seconds() / 3600

    df = df.set_index('timestamp')
    df['trade_freq_7d'] = df.assign(one=1)['one'].rolling('7D').sum()
    df = df.reset_index()

    df['early_exit'] = (df['holding_duration'] < 0.2 * df['holding_duration'].mean()).astype(int)

    plh_n = ((24 - df['post_loss_hours'].fillna(24).clip(0, 24)) / 24).clip(0, 1)
    sz_n  = df['size_deviation'].fillna(0).abs().clip(0, 2) / 2
    df['revenge_score']   = (plh_n + sz_n) / 2.0

    df['emotional_score'] = (
        df['revenge_score'] * 0.4
        + df['size_deviation'].fillna(0).abs().clip(0, 2) / 2 * 0.3
        + df['early_exit'] * 0.3
    )
    df['emotional_state'] = np.select(
        [df['emotional_score'] < 0.33, df['emotional_score'] < 0.66],
        ['calm', 'anxious'],
        default='euphoric'
    )
    return df.fillna(0)

# ─────────────────────────────────────────
# ML MODELS & ADVANCED MODULES
# ─────────────────────────────────────────
FEATURE_COLS = [
    'position_value', 'size_deviation', 'revenge_score',
    'emotional_score', 'post_loss_hours', 'trade_freq_7d',
    'early_exit', 'holding_duration', 'pnl'
]

def run_digital_twin(user_hmm, scaler, feat_cols, base_pnl):
    try:
        sim_X, _       = user_hmm.sample(50)
        sim_X_unscaled = scaler.inverse_transform(sim_X)
        sim_pnl  = sim_X_unscaled[:, feat_cols.index('pnl')] if 'pnl' in feat_cols else np.random.normal(0, 100, 50)
        sim_curve = np.cumsum(sim_pnl) + base_pnl
        return {
            "simulated_equity_curve": sim_curve.tolist(),
            "projected_drawdown": float(np.min(sim_curve) - base_pnl)
        }
    except Exception:
        return None

def run_xai_attribution(X, enriched, feat_cols):
    try:
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_reg.fit(X, enriched['pnl'])
        explainer = shap.TreeExplainer(rf_reg)
        sv = explainer.shap_values(X)

        worst_idx = enriched['pnl'].nsmallest(5).index
        worst_trades_xai = []
        for idx in worst_idx:
            trade_sv     = sv[idx]
            top_features = np.argsort(trade_sv)[:3]
            worst_trades_xai.append({
                "trade_index": int(idx),
                "pnl": float(enriched.loc[idx, 'pnl']),
                "attribution": {feat_cols[i]: float(trade_sv[i]) for i in top_features}
            })
        return worst_trades_xai
    except Exception as e:
        print(f"XAI Error: {e}")
        return []

class Autoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, d))
    def forward(self, x): return self.decoder(self.encoder(x))

def run_models(enriched, feat_cols, X, scaler):
    if len(X) < 4:
        raise ValueError("Insufficient data (need 4+ trades)")

    # GMM Clustering
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    enriched['gmm_cluster'] = gmm.fit_predict(X)
    gmm_sil = silhouette_score(X, enriched['gmm_cluster'])

    # HMM + Digital Twin
    user_hmm = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=42)
    user_hmm.fit(X)
    enriched['hmm_state'] = user_hmm.predict(X)
    digital_twin_data     = run_digital_twin(user_hmm, scaler, feat_cols, enriched['pnl'].sum())

    # Anomaly Detection
    iso       = IsolationForest(contamination=0.1, random_state=42).fit(X)
    iso_score = -iso.score_samples(X)
    thresh    = np.quantile(iso_score, 0.9)
    enriched['anomaly_flag'] = (iso_score > thresh).astype(int)

    # XAI Attribution
    xai_data = run_xai_attribution(X, enriched, feat_cols)

    def label_cluster(idx):
        pts = X[enriched['gmm_cluster'] == idx]
        if len(pts) == 0: return f'cluster_{idx}'
        center = dict(zip(feat_cols, pts.mean(axis=0)))
        if center.get('revenge_score', 0) > 0.5: return 'reactive_emotional'
        if center.get('size_deviation', 0) > 0.5: return 'aggressive_high_activity'
        return 'balanced_tactical'

    enriched['cluster_name'] = enriched['gmm_cluster'].map(
        {i: label_cluster(i) for i in range(4)}
    )
    return enriched, gmm_sil, digital_twin_data, xai_data

# ─────────────────────────────────────────
# RAG & REPORTING
# ─────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def build_rag_index(report):
    client = chromadb.PersistentClient(path=str(BASE_PATH / "chroma_db"))
    emb    = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    coll   = client.get_or_create_collection('session_data', embedding_function=emb)
    docs   = [
        f"Style: {report['behavioral_profile']['trading_style']}, Emotion: {report['behavioral_profile']['dominant_emotion']}.",
        f"Loss aversion: {report['biases']['loss_aversion_lambda']:.2f}. Disposition: {report['biases']['disposition_score']:.2f}.",
        f"Win rate: {report['summary']['win_rate']:.1%}. VaR: {report['risk_profile']['var95']:.3f}."
    ]
    coll.upsert(documents=docs, ids=[f'chunk_{i}' for i in range(len(docs))])

# ─────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────
def execute_pipeline_core(csv_path):
    session_id = str(uuid.uuid4())
    conn       = init_db()

    print("\n[1/7] Loading CSV...")
    df = load_csv(csv_path)
    print(f"      {len(df)} rows loaded | {df['symbol'].nunique()} unique symbols")

    print("\n[2/7] Reconstructing trades (FIFO)...")
    df_realized = reconstruct_trades(df, session_id, conn)

    if df_realized.empty:
        print("⚠️  No matched BUY→SELL pairs found (sells predate this export window).")
        print("    Estimating P&L from NSE/BSE closing prices 30 days before each sell...")
        df_realized = _estimate_pnl_from_market(df, session_id, conn)

    if df_realized.empty:
        raise ValueError("Could not reconstruct any trades. Check your CSV format.")

    print(f"      {len(df_realized)} trades ready")

    print("\n[3/7] Fetching market data (yfinance NSE/BSE)...")
    market_data = fetch_market_data(df_realized)
    print(f"      {len(market_data)}/{df_realized['symbol'].nunique()} symbols enriched")

    print("\n[4/7] Enriching trades with market context...")
    enriched = enrich_trades(df_realized, market_data, session_id, conn)

    if enriched.empty:
        print("⚠️  Market enrichment returned empty — using raw trades with blank market columns.")
        enriched = df_realized.copy()
        enriched['timestamp'] = pd.to_datetime(enriched['timestamp'])
        for col in ['RSI', 'ATR', 'ADX_14', 'SMA50', 'EMA20', 'ATR_Rolling_Mean']:
            enriched[col] = np.nan
        enriched['market_regime'] = 'unknown'

    print(f"      {len(enriched)} trades enriched")

    print("\n[5/7] Computing behavioral features...")
    enriched = compute_all_features(enriched)

    print("\n[6/7] Running ML models (GMM, HMM, Anomaly, XAI)...")
    feat_cols = [c for c in FEATURE_COLS if c in enriched.columns]
    scaler    = StandardScaler()
    X         = scaler.fit_transform(enriched[feat_cols].fillna(0).to_numpy())

    enriched, gmm_sil, dt_data, xai_data = run_models(enriched, feat_cols, X, scaler)
    print(f"      GMM Silhouette: {gmm_sil:.3f}")

    print("\n[7/7] Building report & RAG index...")
    returns = (enriched['pnl'] / enriched['position_value'].replace(0, np.nan)).dropna()
    var95   = float(np.quantile(returns, 0.05)) if not returns.empty else 0.0

    report = {
        'session_id': session_id,
        'summary': {
            'total_trades': len(enriched),
            'win_rate':     float((enriched['pnl'] > 0).mean()),
            'total_pnl':    float(enriched['pnl'].sum()),
        },
        'behavioral_profile': {
            'trading_style':    'disciplined_systematic' if enriched['revenge_score'].mean() < 0.2 else 'reactive_emotional',
            'dominant_cluster': enriched['cluster_name'].value_counts().index[0],
            'dominant_emotion': enriched['emotional_state'].value_counts().index[0],
        },
        'biases': {
            'loss_aversion_lambda': 2.5 if var95 < -0.05 else 1.5,
            'disposition_score':    float((enriched['pnl'] > 0).sum() / max((enriched['pnl'] < 0).sum(), 1)),
            'revenge_trading_rate': float(enriched['revenge_score'].mean()),
            'early_exit_rate':      float(enriched['early_exit'].mean()),
        },
        'risk_profile': {
            'var95':        var95,
            'best_regime':  'trending_bullish',
            'worst_regime': 'risk_off',
        },
        'anomaly': {
            'anomaly_rate':  float(enriched['anomaly_flag'].mean()),
            'anomaly_count': int(enriched['anomaly_flag'].sum()),
        },
        'digital_twin':    dt_data,
        'xai_attribution': xai_data,
    }

    out_report = BASE_PATH / 'behavioral_report.json'
    out_trades = BASE_PATH / 'enriched_trades.csv'
    with open(out_report, 'w') as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
    enriched.to_csv(out_trades, index=False)
    build_rag_index(report)
    conn.close()

    print(f"\n✅ Report saved  → {out_report}")
    print(f"✅ Enriched CSV  → {out_trades}")
    return report

# ─────────────────────────────────────────
# FASTAPI MODULE (REAL-TIME ENGINE)
# ─────────────────────────────────────────
if HAS_FASTAPI:
    app = FastAPI(
        title="BehavioralEdge API Engine",
        description="Real-time webhook ingestion and quantitative analysis."
    )

    class TradeWebhook(BaseModel):
        timestamp: str
        symbol:    str
        side:      str
        quantity:  float
        price:     float

    @app.post("/api/v1/webhook/trade")
    async def receive_trade(trade: TradeWebhook, background_tasks: BackgroundTasks):
        conn = init_db()
        conn.execute(
            "INSERT INTO trades (session_id, timestamp, symbol, side, quantity, price) VALUES (?, ?, ?, ?, ?, ?)",
            ("live_session", trade.timestamp, trade.symbol, trade.side, trade.quantity, trade.price)
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Trade logged. Awaiting daily batch analysis."}

    @app.post("/api/v1/analyze")
    async def analyze_upload(file: UploadFile = File(...)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            report = execute_pipeline_core(tmp_path)
            return {"status": "success", "report": report}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
        finally:
            os.remove(tmp_path)

# ─────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',   help='Path to your trades CSV file')
    parser.add_argument('--serve', action='store_true', help='Launch the FastAPI server')
    args = parser.parse_args()

    if args.serve:
        if not HAS_FASTAPI:
            print("❌ FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")
            sys.exit(1)
        print("🚀 Launching BehavioralEdge Real-Time API Engine on http://0.0.0.0:8000 ...")
        uvicorn.run("pipeline:app", host="0.0.0.0", port=8000, reload=True)
    elif args.csv:
        print("=" * 55)
        print("  BehavioralEdge — Analysis Pipeline")
        print("=" * 55)
        report = execute_pipeline_core(args.csv)
        print("\n" + "=" * 55)
        print(f"  Total trades  : {report['summary']['total_trades']}")
        print(f"  Win rate      : {report['summary']['win_rate']:.1%}")
        print(f"  Total P&L     : {report['summary']['total_pnl']:,.2f}")
        print(f"  Trading style : {report['behavioral_profile']['trading_style']}")
        print(f"  Dom. emotion  : {report['behavioral_profile']['dominant_emotion']}")
        print(f"  Anomalies     : {report['anomaly']['anomaly_count']}")
        print(f"  VaR 95%       : {report['risk_profile']['var95']:.4f}")
        print("=" * 55)
        print("  Pipeline complete! Run: streamlit run app.py")
        print("=" * 55)
    else:
        parser.print_help()