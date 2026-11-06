# Data Dictionary

Dataset: UCI SMS Spam Collection (5,572 messages loaded; 5,157 after cleaning and
de-duplication). Optional own-collected scams can be added in `data/raw/own_scams.csv`.

Roles: **ID** = identifier, **FEATURE** = explanatory attribute, **TARGET** = what we predict,
**DO NOT USE** = must not be given to a model.

## Raw and provenance columns

| Column | Measurement type | Plausible range | Role | Notes |
|--------|------------------|-----------------|------|-------|
| text | Nominal (free text) | 1 to 1000 characters | FEATURE SOURCE | Raw message. All model inputs are derived from it. |
| label | Nominal (binary) | ham / spam | TARGET | spam = scam. Encoded as y = 1 for spam. |
| source | Nominal | uci / own | DO NOT USE | Where the row was collected. Unknown to the user at prediction time. |
| repair_note | Nominal | text | DO NOT USE | Provenance of each repair (html_entity, non_ascii_removed, whitespace_fixed, none). |
| is_long_outlier | Nominal (binary) | 0 / 1 | DO NOT USE | Audit flag, created after cleaning. |
| norm | Nominal (free text) | lower-case text | FEATURE SOURCE | `normalize_text(text)`: URL, phone number, short code and numbers replaced by tokens. |
| y | Nominal (binary) | 0 / 1 | TARGET | Numeric form of label. |

## Hand-built attributes (`src/features.py`)

| Column | Measurement type | Plausible range | Role |
|--------|------------------|-----------------|------|
| msg_length | Ratio | 1 to 1000 | FEATURE |
| word_count | Ratio | 1 to ~200 | FEATURE |
| digit_count | Ratio | 0 to msg_length | FEATURE |
| symbol_count | Ratio | 0 to msg_length | FEATURE |
| digit_ratio | Ratio | 0 to 1 | FEATURE |
| symbol_ratio | Ratio | 0 to 1 | FEATURE |
| upper_ratio | Ratio | 0 to 1 | FEATURE |
| exclaim_count | Ratio | 0 to ~20 | FEATURE |
| has_url | Nominal (binary) | 0 / 1 | FEATURE |
| has_phone_number | Nominal (binary) | 0 / 1 | FEATURE |
| has_shortcode | Nominal (binary) | 0 / 1 | FEATURE |
| has_currency | Nominal (binary) | 0 / 1 | FEATURE |
| length_bucket | Ordinal | short < medium < long (cut points 50 and 120 characters) | FEATURE |

## Text-derived model inputs

| Representation | Used by | Notes |
|----------------|---------|-------|
| Word counts (CountVectorizer, min_df=2) | Naive Bayes | Counts suit the multinomial model. |
| Binary word presence | Decision tree | Presence or absence of a word. |
| TF-IDF | kNN (cosine distance) | Length-normalised, so no extra rescaling is needed. |

## Measurement types present (Lecture 2)

- **Nominal:** text, label, source, has_* flags.
- **Ordinal:** length_bucket.
- **Ratio:** counts, lengths and ratios.
- **Interval:** not present. UCI has no timestamp, so `hour_received` could not be built.
  Three of the four types are present, which meets the requirement.

## Data quality faults found (see `data/clean/audit_report.csv`)

Duplicate records (403), encoding noise (790), inconsistent whitespace (425),
length outliers (68), near-duplicates (34), too-short messages (12),
ALL CAPS casing (107). Conflicting labels, missing text and invalid labels: 0.