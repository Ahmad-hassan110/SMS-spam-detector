# Leakage Statement

## The moment the question is asked

The user pastes one SMS they just received and asks: "Is this a scam?"
At that moment the only thing known is **the message text itself**.

## Columns excluded because they are not known at that moment

| Column | Why it is excluded |
|--------|--------------------|
| label / y | This is the answer being predicted. |
| source | Which dataset a message came from is unknown to the user, and it could let a model
learn the collection method instead of the scam pattern. |
| repair_note | Created by our cleaning step after the fact. |
| is_long_outlier | Created by our audit step after the fact. |
| Any later outcome (for example, "number blocked afterwards", annotator notes) | Only known after the event, not when the message arrives. |

## What the models use

Only `text`, and attributes computed from `text` alone (`normalize_text`, word counts,
TF-IDF). All of these can be computed instantly from a pasted message.

## Other leakage safeguards

- Exact duplicates were removed **before** the train/test split, so the same message
  cannot appear in both.
- The decision threshold was chosen from out-of-fold predictions on the training data,
  not from the test set.
- Hyper-parameters were tuned with cross-validation on the training data only.
- The test set was used once, for the final reported scores.