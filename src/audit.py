import re
import pandas as pd
import numpy as np

RAW = "data/raw/uci_sms.csv" 
CLEAN = "data/clean/sms_clean.csv"

df = pd.read_csv(RAW)
df["source"] = "uci"

raw = df.copy()          
audit = []               

n_dup = int(df.duplicated("text").sum())
audit.append(("Duplicate records", "df.duplicated('text').sum()", n_dup))

g = df.groupby("text")["label"].nunique()
n_conf = int((g > 1).sum())
audit.append(("Conflicting labels", "groupby(text).label.nunique() > 1", n_conf))

n_missing = int((df["text"].isna() | (df["text"].astype(str).str.strip() == "")).sum())
audit.append(("Missing/empty text", "text.isna() | text.strip()==''", n_missing))

n_enc = int(df["text"].astype(str).str.contains(r"&\w+;|[^\x00-\x7F]", regex=True).sum())
audit.append(("Encoding noise", r"regex &\w+; or non-ASCII", n_enc))

n_ws = int(df["text"].astype(str).str.contains(r"^\s|\s$|\s{2,}", regex=True).sum())
audit.append(("Inconsistent whitespace", r"regex ^\s|\s$|\s{2,}", n_ws))

lens = df["text"].astype(str).str.len()
q1, q3 = lens.quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
n_out = int((lens > upper).sum())
audit.append(("Length outliers", f"len > Q3+1.5*IQR ({upper:.0f})", n_out))

n_lab = int((~df["label"].isin(["ham", "spam"])).sum())
audit.append(("Invalid label spelling", "label not in {ham, spam}", n_lab))

def norm_nums(s):
    return re.sub(r"\d+", "<NUM>", str(s).lower().strip())
n_near = int(df["text"].map(norm_nums).duplicated().sum()) - n_dup
audit.append(("Near-duplicates (digits differ)", "duplicated after digits->NUM", max(n_near, 0)))

n_short = int((df["text"].astype(str).str.strip().str.len() <= 3).sum())
audit.append(("Too-short messages", "len(text.strip()) <= 3", n_short))

def mostly_caps(s):
    letters = [c for c in str(s) if c.isalpha()]
    return len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) > 0.8
n_caps = int(df["text"].map(mostly_caps).sum())
audit.append(("Inconsistent casing (ALL CAPS)", "upper letters > 80%", n_caps))

audit_df = pd.DataFrame(audit, columns=["fault", "detected_by", "records_affected"])
print("\n=== AUDIT REPORT ===")
print(audit_df.to_string(index=False))

df["repair_note"] = ""
before = {"rows": len(df), "duplicates": n_dup, "encoding_noise": n_enc,
          "whitespace_issues": n_ws, "conflicts": n_conf, "missing": n_missing}

# 1. label spelling
df["label"] = df["label"].astype(str).str.strip().str.lower()

# 2. missing text drop
mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
df = df[~mask].copy()

# 3. HTML entities
ents = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&nbsp;": " "}
def fix_ent(s):
    for k, v in ents.items():
        s = s.replace(k, v)
    return s
changed = df["text"].map(fix_ent) != df["text"]
df.loc[changed, "repair_note"] += "html_entity;"
df["text"] = df["text"].map(fix_ent)
df["text"] = df["text"].str.replace("£", " GBP ", regex=False).str.replace("€", " EUR ", regex=False)

# 4. non-ASCII garbage hatao
changed = df["text"].str.contains(r"[^\x00-\x7F]", regex=True)
df.loc[changed, "repair_note"] += "non_ascii_removed;"
df["text"] = df["text"].str.replace(r"[^\x00-\x7F]", "", regex=True)

# 5. whitespace
stripped = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
changed = stripped != df["text"]
df.loc[changed, "repair_note"] += "whitespace_fixed;"
df["text"] = stripped

# 6. conflicting labels: dono rows hatao (kis ka label sahi hai pata nahi)
g = df.groupby("text")["label"].nunique()
conf_texts = g[g > 1].index
c = df["text"].isin(conf_texts)
print(f"Conflicting rows dropped: {int(c.sum())}")
df = df[~c].copy()

# 7. exact duplicates hatao (pehla rakho)
df = df.drop_duplicates("text", keep="first").copy()

# 8. empty text dobara check (cleaning ke baad)
df = df[df["text"].str.len() > 0].copy()

# 9. outlier: flag karo, hatao nahi (lambe scams asli ho sakte hain)
lens = df["text"].str.len()
df["is_long_outlier"] = (lens > upper).astype(int)

df["repair_note"] = df["repair_note"].replace("", "none")
df = df.reset_index(drop=True)

# ---------- BEFORE / AFTER ----------
after = {"rows": len(df),
         "duplicates": int(df.duplicated("text").sum()),
         "encoding_noise": int(df["text"].str.contains(r"&\w+;|[^\x00-\x7F]", regex=True).sum()),
         "whitespace_issues": int(df["text"].str.contains(r"^\s|\s$|\s{2,}", regex=True).sum()),
         "conflicts": int((df.groupby("text")["label"].nunique() > 1).sum()),
         "missing": int((df["text"].str.strip() == "").sum())}
ba = pd.DataFrame({"before": before, "after": after})
print("\n=== BEFORE / AFTER ===")
print(ba)

# ---------- VALIDATE ----------
def validate(d):
    t = d["text"].astype(str)
    checks = {
        "1 no null text":            d["text"].notna().all(),
        "2 no null label":           d["label"].notna().all(),
        "3 labels only ham/spam":    d["label"].isin(["ham", "spam"]).all(),
        "4 no empty text":           (t.str.strip().str.len() > 0).all(),
        "5 no duplicate text":       not d.duplicated("text").any(),
        "6 no conflicting labels":   (d.groupby("text")["label"].nunique() <= 1).all(),
        "7 no html entities":        not t.str.contains(r"&\w+;", regex=True).any(),
        "8 no non-ascii":            not t.str.contains(r"[^\x00-\x7F]", regex=True).any(),
        "9 no extra whitespace":     not t.str.contains(r"^\s|\s$|\s{2,}", regex=True).any(),
        "10 text length <= 1000":    (t.str.len() <= 1000).all(),
        "11 spam ratio 5%-50%":      0.05 <= (d["label"] == "spam").mean() <= 0.5,
        "12 at least 2000 rows":     len(d) >= 2000,
    }
    return checks

def run_validate(d, name):
    res = validate(d)
    print(f"\n--- validate() on {name} ---")
    for k, v in res.items():
        print(("PASS " if v else "FAIL ") + k)
    return all(res.values())

ok_raw = run_validate(raw, "RAW")
ok_clean = run_validate(df, "CLEAN")
print("\nRAW passes:", ok_raw, "| CLEAN passes:", ok_clean)

# ---------- SAVE ----------
df.to_csv(CLEAN, index=False)
audit_df.to_csv("data/clean/audit_report.csv", index=False)
ba.to_csv("data/clean/before_after.csv")
print("\nSaved:", CLEAN, "| rows:", len(df))
print(df.label.value_counts())