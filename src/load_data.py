import pandas as pd

df = pd.read_csv("data/raw/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
df.to_csv("data/raw/uci_sms.csv", index=False)

print(df.shape)
print(df.label.value_counts())
print(df.head())