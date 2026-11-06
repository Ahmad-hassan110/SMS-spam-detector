import re
import pandas as pd

CLEAN = "data/clean/sms_clean.csv"
OUT = "data/clean/sms_features.csv"

URL_RE = r"(?:https?://|www\.|\b\w+\.(?:com|net|org|pk|co|info)\b|bit\.ly|tinyurl)"
SHORT_RE = r"\b\d{4,6}\b"          # short codes jaise 80062, 85023
PHONE_RE = r"\b(?:\+?\d{10,13}|0\d{2,3}[- ]?\d{7})\b"
CURR_RE = r"(?:gbp|eur|\$|usd|pkr|\brs\.?\s?\d|\bcash\b)"

TOK_URL = r"(?:https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|pk|co|info)\b\S*|bit\.ly\S*|tinyurl\S*)"

def normalize_text(s):
    s = str(s).lower()
    s = re.sub(TOK_URL, " urltoken ", s)
    s = re.sub(r"\+?\d{10,13}|\b0\d{2,3}[- ]?\d{7}\b", " phonetoken ", s)
    s = re.sub(r"\b\d{4,6}\b", " shortcodetoken ", s)
    s = re.sub(r"\d+", " numtoken ", s)
    return re.sub(r"\s+", " ", s).strip()

def make_features(df):
    t = df["text"].astype(str)
    f = pd.DataFrame(index=df.index)
    f["msg_length"] = t.str.len()
    f["word_count"] = t.str.split().str.len()
    f["digit_count"] = t.str.count(r"\d")
    f["symbol_count"] = t.str.count(r"[^A-Za-z0-9\s]")
    f["digit_ratio"] = f["digit_count"] / f["msg_length"]
    f["symbol_ratio"] = f["symbol_count"] / f["msg_length"]
    f["upper_ratio"] = t.str.count(r"[A-Z]") / f["msg_length"]
    f["exclaim_count"] = t.str.count("!")
    f["has_url"] = t.str.contains(URL_RE, case=False, regex=True).astype(int)
    f["has_phone_number"] = t.str.contains(PHONE_RE, regex=True).astype(int)
    f["has_shortcode"] = t.str.contains(SHORT_RE, regex=True).astype(int)
    f["has_currency"] = t.str.contains(CURR_RE, case=False, regex=True).astype(int)
    f["length_bucket"] = pd.cut(f["msg_length"], bins=[0, 50, 120, 10000],
                                labels=["short", "medium", "long"])
    return f

if __name__ == "__main__":
    df = pd.read_csv(CLEAN)
    feats = make_features(df)
    out = pd.concat([df[["text", "label", "source"]], feats], axis=1)
    out.to_csv(OUT, index=False)
    print(out.shape)
    print(feats.describe().T[["mean", "min", "max"]])
    print("\nFeature mean by class (scam vs ham):")
    print(out.groupby("label")[["msg_length", "digit_ratio", "upper_ratio",
          "has_url", "has_phone_number", "has_shortcode", "has_currency"]].mean().T)