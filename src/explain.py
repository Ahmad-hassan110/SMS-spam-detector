import re
import joblib
import numpy as np
from features import normalize_text

LEXICON = {
    "urgent": ("Urgency", "Jaldi karwane ki koshish, taake aap soch na saken."),
    "immediately": ("Urgency", "Jaldi karwane ki koshish, taake aap soch na saken."),
    "now": ("Urgency", "Foran action ka dabao."),
    "expires": ("Urgency", "Waqt khatam hone ka dar."),
    "won": ("Prize bait", "Inaam ka lalach de kar phansana."),
    "prize": ("Prize bait", "Inaam ka lalach de kar phansana."),
    "winner": ("Prize bait", "Inaam ka lalach de kar phansana."),
    "claim": ("Prize bait", "Inaam wasool karne ke naam par maloomat ya paisa maangna."),
    "free": ("Prize bait", "Muft cheez ka lalach."),
    "cash": ("Prize bait", "Paison ka lalach."),
    "click": ("Fake link", "Jali website par le jane ki koshish."),
    "urltoken": ("Fake link", "Message mein link hai, jali website ho sakti hai."),
    "verify": ("Account scam", "Bank ban kar aap ki maloomat nikalna."),
    "blocked": ("Account scam", "Account band hone ka dar dikha kar maloomat maangna."),
    "otp": ("Account scam", "OTP kisi ko na dein, bank kabhi nahi maangta."),
    "pin": ("Account scam", "PIN kisi ko na dein."),
    "phonetoken": ("Callback trick", "Phone number par call karwa kar paisa katna."),
    "shortcodetoken": ("Premium short code", "Short code par reply karne se paise kat sakte hain."),
    "gbp": ("Money lure", "Paison ka lalach."),
    "parcel": ("Fake delivery", "Jali delivery fees maangna."),
    "delivery": ("Fake delivery", "Jali delivery fees maangna."),
    "customs": ("Fake delivery", "Jali customs fees maangna."),
}

def explain(text, model_name="nb", top=5):
    m = joblib.load(f"models/{model_name}.joblib")
    norm = normalize_text(text)
    p = float(m.predict_proba([norm])[0, 1])
    reasons = []
    if model_name == "nb":
        vec, clf = m.named_steps["vec"], m.named_steps["clf"]
        lr = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        vocab = {w: i for i, w in enumerate(vec.get_feature_names_out())}
        for w in set(norm.split()):
            if w in vocab and lr[vocab[w]] > 0.5:
                reasons.append((w, float(lr[vocab[w]])))
        reasons.sort(key=lambda x: -x[1])
    else:
        for w in set(norm.split()):
            if w in LEXICON:
                reasons.append((w, 1.0))
    out = []
    for w, s in reasons[:top]:
        cat, why = LEXICON.get(w, ("Suspicious word", "Ye lafz scam messages mein aksar aata hai."))
        out.append({"word": w, "score": s, "trick": cat, "why": why})
    return p, out

if __name__ == "__main__":
    for msg in ["URGENT! You have won a prize. Claim now at bit.ly/abc123 or call 03001234567",
                "Ok see you at dinner tonight"]:
        p, r = explain(msg)
        print(f"\n{msg}\nScam probability: {p:.2%}")
        for x in r:
            print(f"  - {x['word']} ({x['trick']}): {x['why']}")