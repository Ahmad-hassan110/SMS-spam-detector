import os, sys, re, html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.append(os.path.join(BASE, "src"))
from features import normalize_text, TOK_URL
from explain import explain

st.set_page_config(page_title="SMS Scam Shield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
/* page */
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: 0.3px;}
h2 {border-left: 5px solid #00C2A8; padding-left: 12px;}

/* sidebar */
/* sidebar nav buttons */
[data-testid="stSidebar"] [role="radiogroup"] {gap: 10px;}

[data-testid="stSidebar"] [role="radiogroup"] > label {
    background: #131C2E;
    border: 1px solid #22304D;
    border-radius: 12px;
    padding: 12px 16px;
    width: 100%;
    cursor: pointer;
    transition: all .18s ease;
}

/* gol circle chhupao */
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {display: none;}

[data-testid="stSidebar"] [role="radiogroup"] > label p {
    font-size: 0.98rem; font-weight: 600; color: #B8C4DA; margin: 0;
}

/* hover */
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: #1A2742;
    border-color: #00C2A8;
    transform: translateX(4px);
}

/* selected */
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, #00C2A8, #0090D0);
    border-color: transparent;
}

/* sidebar title */
[data-testid="stSidebar"] h1 {font-size: 1.4rem; margin-bottom: 4px;}
[data-testid="stSidebar"] hr {border-color: #1F2B44;}

/* metric cards */
[data-testid="stMetric"] {
    background: #131C2E;
    border: 1px solid #22304D;
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="stMetricValue"] {color: #00E0C0;}

/* buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #00C2A8, #0090D0);
    color: #fff; border: none; border-radius: 10px;
    padding: 0.55rem 1.3rem; font-weight: 600;
    box-shadow: none !important;
    transition: none;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stButton > button:focus, .stDownloadButton > button:focus,
.stButton > button:active, .stDownloadButton > button:active {
    background: linear-gradient(90deg, #00C2A8, #0090D0);
    color: #fff;
    transform: none;
    box-shadow: none !important;
    outline: none;
}

/* text areas, inputs */
textarea, input {border-radius: 10px !important;}
[data-testid="stTextArea"] textarea {background: #131C2E; border: 1px solid #22304D;}

/* tables and images */
[data-testid="stDataFrame"] {border: 1px solid #22304D; border-radius: 12px; overflow: hidden;}
[data-testid="stImage"] img {border-radius: 12px; background: #fff; padding: 6px;}

/* hero banner */
.hero {
    background: linear-gradient(120deg, #0E2A47 0%, #0B3B3A 100%);
    border: 1px solid #1F3B5C; border-radius: 16px;
    padding: 22px 28px; margin-bottom: 22px;
}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: 6px 0 0 0; color: #A9B8D0;}

/* verdict boxes */
.verdict {border-radius: 14px; padding: 16px 22px; font-size: 1.5rem; font-weight: 700; margin: 10px 0;}
.verdict.scam {background: rgba(255,82,82,0.15); border: 1px solid #FF5252; color: #FF8A80;}
.verdict.safe {background: rgba(0,200,120,0.15); border: 1px solid #00C878; color: #69F0AE;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ SMS Scam Shield</h1>
  <p>Paste a message. Find out if it is a scam, and learn the trick behind it.</p>
</div>
""", unsafe_allow_html=True)

NAMES = {"nb": "Naive Bayes", "tree": "Decision Tree", "knn": "kNN"}

@st.cache_resource
def get_model(n):
    return joblib.load(f"models/{n}.joblib")

@st.cache_data
def get_test():
    return pd.read_csv("data/clean/test.csv")

@st.cache_data
def test_probs(n):
    t = get_test()
    return get_model(n).predict_proba(t["norm"])[:, 1]

def best_thr(n):
    return float(joblib.load(f"models/{n}_threshold.joblib"))

# ---------- session defaults ----------
ss = st.session_state
ss.setdefault("model", "nb")
ss.setdefault("thr", best_thr("nb"))
ss.setdefault("cost_fn", 10)
ss.setdefault("cost_fp", 1)

def use_optimal():
    ss["thr"] = best_thr(ss["model"])

# ---------- sidebar ----------
st.sidebar.title("🛡️ SMS Scam Shield")
page = st.sidebar.radio("Screen", [
    "1. Data & quality", "2. Model performance", "3. Single prediction",
    "4. Batch scoring", "5. Settings", "6. Learn the tricks"])
st.sidebar.caption(f"Model: {NAMES[ss['model']]} | Threshold: {ss['thr']:.2f}")

PAT = {"urltoken": TOK_URL,
       "phonetoken": r"\+?\d{10,13}|\b0\d{2,3}[- ]?\d{7}\b",
       "shortcodetoken": r"\b\d{4,6}\b",
       "numtoken": r"\d+"}

def highlight(text, items):
    spans = []
    for it in items:
        pat = PAT.get(it["word"], r"\b" + re.escape(it["word"]) + r"\b")
        for m in re.finditer(pat, text, flags=re.I):
            spans.append((m.start(), m.end(), it["trick"]))
    spans.sort()
    out, pos = "", 0
    for s, e, t in spans:
        if s < pos:
            continue
        out += html.escape(text[pos:s])
        out += (f'<mark style="background:#ffd54f;color:#111;padding:2px 6px;border-radius:5px;font-weight:600" '
        f'title="{t}">{html.escape(text[s:e])}</mark>')
        pos = e
    return out + html.escape(text[pos:])

def verdict(p, thr):
    return "🚨 SCAM" if p >= thr else "✅ Looks safe"

# ================= 1. DATA & QUALITY =================
if page.startswith("1"):
    st.header("Data & quality")
    st.subheader("Audit report (faults found)")
    st.dataframe(pd.read_csv("data/clean/audit_report.csv"), use_container_width=True)
    st.subheader("Before / after repair")
    st.dataframe(pd.read_csv("data/clean/before_after.csv", index_col=0), use_container_width=True)
    clean = pd.read_csv("data/clean/sms_clean.csv")
    st.subheader(f"Cleaned data ({len(clean)} rows)")
    st.dataframe(clean.head(50), use_container_width=True)
    st.download_button("⬇️ Download cleaned CSV", clean.to_csv(index=False).encode(),
                       "sms_clean.csv", "text/csv")

# ================= 2. MODEL PERFORMANCE =================
elif page.startswith("2"):
    st.header("Model performance")
    comp = pd.read_csv("data/clean/model_comparison.csv").drop(columns="best_params")
    comp["model"] = comp["model"].map(NAMES)
    st.subheader("Comparison (test set, baseline beside every accuracy)")
    st.dataframe(comp.round(4), use_container_width=True)
    st.caption("Baseline = accuracy of always saying 'ham'. CV F1 shown as mean and std (5-fold, train only).")
    st.subheader("Threshold & cost results")
    thr_df = pd.read_csv("data/clean/threshold_results.csv")
    thr_df["model"] = thr_df["model"].map(NAMES)
    st.dataframe(thr_df.round(4), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image("docs/roc.png", caption="ROC curves with AUC")
    if os.path.exists("docs/alpha_vs_f1.png"):
        c2.image("docs/alpha_vs_f1.png", caption="Naive Bayes: alpha vs F1")
    st.subheader("Confusion matrix")
    n = st.selectbox("Model", list(NAMES), format_func=NAMES.get, index=list(NAMES).index(ss["model"]))
    t = get_test()
    p = test_probs(n)
    thr = best_thr(n)
    tn, fp, fn, tp = confusion_matrix(t["y"], (p >= thr).astype(int)).ravel()
    st.write(f"Threshold used: **{thr:.2f}** (cost-optimal for this model)")
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
                 index=["Actual ham", "Actual scam"],
                 columns=["Predicted ham", "Predicted scam"]))

# ================= 3. SINGLE PREDICTION =================
elif page.startswith("3"):
    st.header("Check one message")
    msg = st.text_area("Paste the SMS here", height=140,
        placeholder="e.g. URGENT! Your JazzCash account is blocked. Verify now at bit.ly/xyz")
    if st.button("Check message", type="primary") and msg.strip():
        p, reasons = explain(msg, ss["model"])
        is_scam = p >= ss["thr"]
        st.markdown(
          f'<div class="verdict {"scam" if is_scam else "safe"}">{verdict(p, ss["thr"])}</div>',
          unsafe_allow_html=True)
        st.metric("Scam probability", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        if reasons:
            st.markdown("**Why it looks suspicious:**")
            for r in reasons:
                st.write(f"- **{r['word']}** ({r['trick']}): {r['why']}")
        else:
            st.write("No suspicious words found.")
        st.caption("Tip: see the highlighted version on screen 6.")

# ================= 4. BATCH SCORING =================
elif page.startswith("4"):
    st.header("Batch scoring")
    st.write("Upload a CSV with a column named **text**.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            d = pd.read_csv(up)
        except Exception:
            up.seek(0)
            d = pd.read_csv(up, engine="python", on_bad_lines="skip")
            st.warning("Some malformed lines were skipped. Put each message in double quotes if it contains a comma.")
        if "text" not in d.columns:
            st.error("CSV must have a 'text' column.")
        else:
            d["text"] = d["text"].astype(str)
            m = get_model(ss["model"])
            d["scam_probability"] = m.predict_proba(d["text"].map(normalize_text))[:, 1].round(4)
            d["verdict"] = np.where(d["scam_probability"] >= ss["thr"], "SCAM", "OK")
            d = d.sort_values("scam_probability", ascending=False).reset_index(drop=True)
            d.insert(0, "rank", d.index + 1)
            c1, c2 = st.columns(2)
            only = c1.checkbox("Show only SCAM", value=False)
            minp = c2.slider("Minimum probability", 0.0, 1.0, 0.0, 0.01)
            view = d[d["scam_probability"] >= minp]
            if only:
                view = view[view["verdict"] == "SCAM"]
            st.write(f"{len(view)} of {len(d)} messages shown, {int((d['verdict']=='SCAM').sum())} flagged as scam.")
            st.dataframe(view, use_container_width=True)
            st.download_button("⬇️ Download result", view.to_csv(index=False).encode(),
                               "scored.csv", "text/csv")

# ================= 5. SETTINGS =================
elif page.startswith("5"):
    st.header("Settings")
    st.selectbox("Model", list(NAMES), format_func=NAMES.get, key="model")
    c1, c2 = st.columns(2)
    c1.number_input("Cost of MISSING a scam", 1, 1000, key="cost_fn")
    c2.number_input("Cost of a FALSE ALARM", 1, 1000, key="cost_fp")
    st.slider("Decision threshold", 0.01, 0.99, step=0.01, key="thr")
    st.button("Set cost-optimal threshold (from training data)", on_click=use_optimal)

    t = get_test()
    p = test_probs(ss["model"])
    y = t["y"]
    def stats(th):
        tn, fp, fn, tp = confusion_matrix(y, (p >= th).astype(int)).ravel()
        return tn, fp, fn, tp, fn * ss["cost_fn"] + fp * ss["cost_fp"]
    tn, fp, fn, tp, cost = stats(ss["thr"])
    acc = (tn + tp) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    base = 1 - y.mean()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Expected cost", cost)
    m2.metric("Accuracy", f"{acc:.3f}", f"baseline {base:.3f}", delta_color="off")
    m3.metric("Precision", f"{prec:.3f}")
    m4.metric("Recall (scams caught)", f"{rec:.3f}")
    m5.metric("Missed scams", int(fn))
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
                 index=["Actual ham", "Actual scam"],
                 columns=["Predicted ham", "Predicted scam"]))
    ths = np.round(np.linspace(0.01, 0.99, 99), 2)
    curve = pd.DataFrame({"threshold": ths, "cost": [stats(x)[4] for x in ths]}).set_index("threshold")
    st.subheader("Cost vs threshold")
    st.line_chart(curve)

# ================= 6. LEARN THE TRICKS =================
else:
    st.header("Learn to spot the trick")
    msg = st.text_area("Paste a message", height=120,
        value="URGENT! You have won a prize. Claim now at bit.ly/abc123 or call 03001234567")
    if msg.strip():
        p, reasons = explain(msg, "nb", top=10)
        st.markdown(f"**Scam probability: {p:.1%}**")
        st.markdown(f'<div style="font-size:18px;line-height:1.8">{highlight(msg, reasons)}</div>',
                    unsafe_allow_html=True)
        st.markdown("---")
        if reasons:
            st.markdown("**What each highlighted word means:**")
            seen = set()
            for r in reasons:
                if r["word"] in seen:
                    continue
                seen.add(r["word"])
                st.write(f"🟨 **{r['word']}** → *{r['trick']}*: {r['why']}")
        else:
            st.success("No suspicious words found in this message.")