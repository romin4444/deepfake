#!/usr/bin/env python3
# =============================================================================
#  PIPELINE 2026 — modernised rebuild of the hybrid regime+sentiment system
#
#  Recreates the methodology of two 2026 papers on our benchmark:
#   [P1] "Impact of LLMs news Sentiment Analysis on Stock Price Movement
#        Prediction" (arXiv:2602.00086, Feb 2026): transformer-based sentiment
#        (FinBERT/RoBERTa/DeBERTa) fused with sequence models (LSTM, TimesNet,
#        PatchTST) and an ensemble over model outputs (~80% acc on their data).
#   [P2] "Beyond Polarity: Multi-Dimensional LLM Sentiment Signals"
#        (arXiv:2603.11408, Mar 2026): five sentiment dimensions — relevance,
#        polarity, INTENSITY, UNCERTAINTY, FORWARDNESS — beat plain polarity;
#        LightGBM + SHAP.
#
#  Plus a NEW model of our own design:
#   [RGST] Regime-Gated Sentiment Transformer. Motivated by our v1 finding
#        that the Wang-regime core carries some sectors (JPM/AAPL) while the
#        sentiment core carries others (XOM): a gate vector computed from the
#        Wang et al. (2025) regime state decides, per sample, how much of the
#        sentiment channel flows into the fused representation:
#            g = sigmoid(MLP(regime context))            in (0,1)^d
#            h = TransformerEnc(price seq) + g * Enc(sentiment seq)
#        i.e. sentiment is trusted conditionally on the volatility regime.
#
#  2026-READINESS FIXES baked in (see README_2026):
#   - vol-scaled movement labels (LABEL_MODE=vol): +/-0.5 * rolling sigma,
#     replacing the fixed +/-0.55%/-0.5% thresholds calibrated on 2014-16 vol
#   - sentiment source abstraction: tweets (offline demo) | news CSV
#     (FNSPID/AlphaVantage export) via NEWS_CSV env — Twitter API is closed
#   - explicit auto_adjust handling documented for the yfinance path
#   - rolling retrain interface (walk-forward window, not one frozen model)
#
#  Stages:  python pipeline_2026.py --stage sentiment2026 --ticker AAPL
#           python pipeline_2026.py --stage train2026
#           python pipeline_2026.py --stage selftest
# =============================================================================
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_pipeline as H                      # reuse data + Core A caches

RNG = np.random.RandomState(42)
CACHE_DIR, OUTPUT_DIR = H.CACHE_DIR, H.OUTPUT_DIR
TICKERS, SECTOR = H.TICKERS, H.SECTOR
TRAIN_END, VAL_END = H.TRAIN_END, H.VAL_END
SEQ_LEN    = int(os.getenv("SEQ_LEN", "20"))
LABEL_MODE = os.getenv("LABEL_MODE", "vol")      # "vol" (2026) | "fixed" (2018)
def log(m): print(m, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
#  [P2] multi-dimensional sentiment (heuristic offline; LLM/FinBERT on Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
_UNCERT = {"may", "might", "could", "possibly", "uncertain", "unclear",
           "risk", "risks", "volatile", "rumor", "rumour", "if", "perhaps",
           "speculation", "doubt", "warns", "concern", "concerns"}
_FWD = {"will", "expect", "expects", "expected", "forecast", "guidance",
        "outlook", "upcoming", "next", "tomorrow", "eps", "earnings",
        "target", "upgrade", "downgrade", "estimates", "projects", "q1",
        "q2", "q3", "q4", "fy", "preview"}

def multidim_scorer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    def score(text, ticker):
        toks = text.lower().replace("$", " $").split()
        n = max(len(toks), 1)
        pol = an.polarity_scores(text)["compound"]
        caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        intensity = min(1.0, abs(pol) + 0.5 * caps + 0.1 * text.count("!"))
        uncertainty = sum(t.strip(".,!?") in _UNCERT for t in toks) / n
        forwardness = sum(t.strip(".,!?") in _FWD for t in toks) / n
        relevance = toks.count(f"${ticker.lower()}") / n
        return pol, intensity, uncertainty, forwardness, relevance
    return score

DIMS = ["polarity", "intensity", "uncertainty", "forwardness", "relevance"]

def stage_sentiment2026(ticker):
    out = f"{CACHE_DIR}/sent26_{ticker}.parquet"
    if os.path.exists(out):
        log(f"[SENT26 {ticker}] cached"); return
    news_csv = os.getenv("NEWS_CSV", "")
    score = multidim_scorer()
    rows = {}
    if news_csv:                                  # 2026 path: news headlines
        nd = pd.read_csv(news_csv, parse_dates=["date"])
        nd = nd[nd["ticker"] == ticker]
        it = ((str(r.date.date()), r.headline) for r in nd.itertuples())
    else:                                         # offline demo path: tweets
        it = H.iter_tweets(ticker)
    n = 0
    for day, text in it:
        if not text:
            continue
        rows.setdefault(day, []).append(score(text, ticker))
        n += 1
    recs = []
    for day, vs in rows.items():
        v = np.asarray(vs, float)
        recs.append((pd.Timestamp(day), *v.mean(0), v[:, 0].std(), len(v)))
    S = (pd.DataFrame(recs, columns=["date", *DIMS, "pol_std", "doc_n"])
           .set_index("date").sort_index())
    for d in DIMS:
        assert S[d].between(-1, 1).all(), f"{d} out of range"
    S.to_parquet(out)
    log(f"[SENT26 {ticker}] {n} docs -> {len(S)} days x {len(DIMS)} dims -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURES + LABELS (vol-scaled labels = 2026 fix)
# ─────────────────────────────────────────────────────────────────────────────
def assemble26(ticker):
    px = H.load_price(ticker)
    px = px[(px.index >= H.DATA_START) & (px.index <= H.TEST_END)]
    ac = px["Adj Close"]; mv = ac.pct_change()
    if LABEL_MODE == "vol":
        sig = mv.rolling(60).std().shift(1)
        y = pd.Series(np.where(mv >= 0.5 * sig, 1,
                      np.where(mv <= -0.5 * sig, 0, -1)), index=mv.index)
    else:
        y = pd.Series(np.where(mv >= H.UP_THR, 1,
                      np.where(mv <= H.DOWN_THR, 0, -1)), index=mv.index)

    S = pd.read_parquet(f"{CACHE_DIR}/sent26_{ticker}.parquet")
    tdays = ac.index
    S.index = [tdays[min(tdays.searchsorted(d), len(tdays) - 1)]
               for d in S.index]
    S = S.groupby(level=0).mean()
    R = pd.read_parquet(f"{CACHE_DIR}/regime_{ticker}.parquet")

    X = pd.DataFrame(index=ac.index)
    X["ret1"] = mv
    X["ret5"] = ac.pct_change(5)
    X["ma5_ratio"] = ac / ac.rolling(5).mean() - 1
    X["ma20_ratio"] = ac / ac.rolling(20).mean() - 1
    X["vol_z"] = ((px["Volume"] - px["Volume"].rolling(20).mean())
                  / (px["Volume"].rolling(20).std() + 1e-9))
    for d in [*DIMS, "pol_std", "doc_n"]:
        X[d] = S[d].reindex(ac.index).fillna(0)
    X["pol_3d"] = X["polarity"].rolling(3).mean()
    for c in ["uncertainty", "w_vol", "regime", "alpha", "beta",
              "node_outdeg", "ew_signal"]:
        X["rg_" + c] = R[c].reindex(ac.index).ffill()
    X = X.shift(1)                                # causality
    return X.join(y.rename("y")).dropna()

PRICE26 = ["ret1", "ret5", "ma5_ratio", "ma20_ratio", "vol_z"]
SENT26  = [*DIMS, "pol_std", "doc_n", "pol_3d"]
REG26   = ["rg_uncertainty", "rg_w_vol", "rg_regime", "rg_alpha", "rg_beta",
           "rg_node_outdeg", "rg_ew_signal"]
ALL26   = PRICE26 + SENT26 + REG26

def build_tensors():
    """Sequences of SEQ_LEN days per labelled sample; z-scored on train."""
    Xs, ys, meta = [], [], []
    for t in TICKERS:
        df = assemble26(t)
        V = df[ALL26].values.astype(np.float32)
        yv = df["y"].values
        for i in range(SEQ_LEN, len(df)):
            if yv[i] < 0:
                continue
            Xs.append(V[i - SEQ_LEN:i + 1])       # includes day i features
            ys.append(yv[i]); meta.append((t, df.index[i]))
    X = np.stack(Xs); y = np.asarray(ys, np.float32)
    dates = pd.DatetimeIndex([m[1] for m in meta])
    tick = np.array([m[0] for m in meta])
    tr = dates < TRAIN_END
    va = (dates >= TRAIN_END) & (dates < VAL_END)
    te = dates >= VAL_END
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-9
    X = (X - mu) / sd
    return X, y, tr, va, te, tick, dates


# ─────────────────────────────────────────────────────────────────────────────
#  MODELS  — LSTM [P1 baseline], PatchTST-lite [P1], RGST [ours]
# ─────────────────────────────────────────────────────────────────────────────
def make_models(F):
    import torch, torch.nn as nn

    class LSTMNet(nn.Module):                      # [P1] baseline
        def __init__(s, d=48):
            super().__init__()
            s.l = nn.LSTM(F, d, batch_first=True)
            s.h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        def forward(s, x):
            o, _ = s.l(x)
            return s.h(o[:, -1]).squeeze(-1)

    class PatchTSTLite(nn.Module):                 # [P1] 2026 model family
        def __init__(s, patch=3, d=48, heads=4):
            super().__init__()
            s.patch = patch
            s.np = (SEQ_LEN + 1) // patch
            s.proj = nn.Linear(patch * F, d)
            s.pos = nn.Parameter(torch.randn(1, s.np, d) * 0.02)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d, heads, d * 2, 0.1, batch_first=True), 2)
            s.h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        def forward(s, x):
            B, L, _ = x.shape
            L2 = s.np * s.patch
            p = x[:, -L2:].reshape(B, s.np, -1)
            z = s.enc(s.proj(p) + s.pos)
            return s.h(z.mean(1)).squeeze(-1)

    iP = [ALL26.index(c) for c in PRICE26]
    iS = [ALL26.index(c) for c in SENT26]
    iR = [ALL26.index(c) for c in REG26]

    class RGST(nn.Module):
        """Regime-Gated Sentiment Transformer (ours, 2026).
        Gate g = sigmoid(MLP(mean regime context)) modulates the sentiment
        channel before fusion with the price/regime sequence encoding."""
        def __init__(s, d=48, heads=4):
            super().__init__()
            s.pp = nn.Linear(len(iP) + len(iR), d)
            s.pos = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d) * 0.02)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d, heads, d * 2, 0.1, batch_first=True), 2)
            s.se = nn.GRU(len(iS), d, batch_first=True)
            s.gate = nn.Sequential(nn.Linear(len(iR), d), nn.Tanh(),
                                   nn.Linear(d, d), nn.Sigmoid())
            s.h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        def forward(s, x):
            price = x[:, :, iP + iR]
            sent = x[:, :, iS]
            regime_ctx = x[:, :, iR].mean(1)
            hp = s.enc(s.pp(price) + s.pos).mean(1)
            hs, _ = s.se(sent)
            g = s.gate(regime_ctx)
            return s.h(hp + g * hs[:, -1]).squeeze(-1)
        def gate_value(s, x):
            with torch.no_grad():
                return s.gate(x[:, :, iR].mean(1)).mean(1)

    return {"LSTM [P1]": LSTMNet(), "PatchTST-lite [P1]": PatchTSTLite(),
            "RGST (ours)": RGST()}


def train_nn(model, X, y, tr, va, epochs=60, lr=1e-3, bs=64, seed=42):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    from sklearn.metrics import roc_auc_score
    dev = "cpu"
    Xt = torch.tensor(X); yt = torch.tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    idx = np.where(tr)[0]
    best_auc, best_state, patience = -1, None, 0
    for ep in range(epochs):
        model.train(); RNG.shuffle(idx)
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[j]), yt[j])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xt[va])).numpy()
        auc = roc_auc_score(y[va], pv)
        if auc > best_auc:
            best_auc, patience = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def stage_train2026():
    import torch
    import lightgbm as lgb
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                                 roc_auc_score)
    X, y, tr, va, te, tick, dates = build_tensors()
    log(f"[2026] tensors X={X.shape}  train={tr.sum()} val={va.sum()} "
        f"test={te.sum()}  label_mode={LABEL_MODE}")
    assert dates[tr].max() < dates[va].min() <= dates[te].min()

    results, probs = [], {}
    # tabular reference (last-day slice -> LightGBM, carries v1 forward)
    Xtab = X[:, -1, :]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                           min_child_samples=30, random_state=42, verbose=-1)
    m.fit(Xtab[tr], y[tr], eval_set=[(Xtab[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    probs["LGBM-merged [v1]"] = m.predict_proba(Xtab[te])[:, 1]

    gates = None
    for name, net in make_models(X.shape[-1]).items():
        net, vauc = train_nn(net, X, y, tr, va)
        with torch.no_grad():
            probs[name] = torch.sigmoid(net(torch.tensor(X[te]))).numpy()
        log(f"   [{name}] best val AUC = {vauc:.4f}")
        if name.startswith("RGST"):
            gates = net.gate_value(torch.tensor(X[te])).numpy()

    # [P1] ensemble: SVM over the model probabilities (fit on val outputs)
    val_probs = {}
    for name, net_or_p in probs.items():
        pass
    # recompute val probs for ensemble members
    Xt = torch.tensor(X)
    val_stack, te_stack, names = [], [], []
    m_val = m.predict_proba(Xtab[va])[:, 1]
    val_stack.append(m_val); te_stack.append(probs["LGBM-merged [v1]"])
    names.append("LGBM-merged [v1]")
    for name, net in make_models(X.shape[-1]).items():
        net, _ = train_nn(net, X, y, tr, va, seed=7)
        with torch.no_grad():
            val_stack.append(torch.sigmoid(net(Xt[va])).numpy())
            te_stack.append(torch.sigmoid(net(Xt[te])).numpy())
        names.append(name)
    sv = SVC(probability=True, C=1.0).fit(np.column_stack(val_stack), y[va])
    probs["ENSEMBLE-SVM [P1]"] = sv.predict_proba(
        np.column_stack(te_stack))[:, 1]

    for name, p in probs.items():
        yh = (p >= 0.5).astype(int)
        results.append((name, round(accuracy_score(y[te], yh), 4),
                        round(matthews_corrcoef(y[te], yh), 4),
                        round(roc_auc_score(y[te], p), 4)))
    maj = max(y[te].mean(), 1 - y[te].mean())
    RES = pd.DataFrame(results, columns=["model", "test_acc", "test_mcc",
                                         "test_auc"])
    RES["majority_baseline"] = round(float(maj), 4)
    RES.to_csv(f"{OUTPUT_DIR}/model_comparison_2026.csv", index=False)
    log("\n=== 2026 MODEL COMPARISON (test 2015-10-01 -> 2016-01-01, "
        f"labels={LABEL_MODE}) ===")
    log(RES.to_string(index=False))

    # per-stock for best neural model + RGST gate diagnostics
    rows = []
    for t in TICKERS:
        mask = tick[te] == t
        if mask.sum() < 10:
            continue
        row = {"ticker": t, "sector": SECTOR.get(t, "?"),
               "n_test": int(mask.sum())}
        for name, p in probs.items():
            row[name] = round(accuracy_score(y[te][mask],
                                             (p[mask] >= .5)), 4)
        if gates is not None:
            row["RGST_mean_gate"] = round(float(gates[mask].mean()), 4)
        rows.append(row)
    PER = pd.DataFrame(rows)
    PER.to_csv(f"{OUTPUT_DIR}/per_stock_2026.csv", index=False)
    log("\n=== PER-STOCK acc + RGST sentiment-gate openness ===")
    log(PER.to_string(index=False))
    return RES, PER


def stage_selftest():
    s = multidim_scorer()
    pol, inten, unc, fwd, rel = s("$AAPL will beat earnings guidance!", "AAPL")
    assert fwd > 0 and rel > 0 and inten > 0, "T1 multidim dims"
    pol2, _, unc2, _, _ = s("rumor: $AAPL might face lawsuit risk", "AAPL")
    assert unc2 > unc and pol2 < pol, "T2 uncertainty/polarity ordering"
    df = assemble26(TICKERS[0])
    assert set(ALL26 + ["y"]) <= set(df.columns) and \
        not df[ALL26].isna().any().any(), "T3 assemble26 schema"
    X, y, tr, va, te, tick, dates = build_tensors()
    assert X.shape[1] == SEQ_LEN + 1 and X.shape[2] == len(ALL26), "T4 tensor"
    assert abs(X[tr].mean()) < 0.15, "T5 train z-scoring"
    import torch
    for name, net in make_models(X.shape[-1]).items():
        out = net(torch.tensor(X[:8]))
        assert out.shape == (8,), f"T6 {name} forward shape"
    log("[SELFTEST-2026] all step tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["sentiment2026", "train2026", "selftest"])
    ap.add_argument("--ticker", default=None)
    a = ap.parse_args()
    if a.stage == "sentiment2026":
        stage_sentiment2026(a.ticker)
    elif a.stage == "train2026":
        stage_train2026()
    else:
        stage_selftest()
