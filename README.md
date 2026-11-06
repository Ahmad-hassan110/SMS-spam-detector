# 🛡️ SMS Scam and Fraud Shield

An interactive Machine Learning web application that helps users identify suspicious SMS messages and understand common scam techniques.

🚀 **https://sms-spam-detector-777.streamlit.app**

## Overview

SMS Scam and Fraud Shield analyzes text messages and provides:

- **Scam Detection:** Classifies messages as Scam or Safe.
- **Scam Probability:** Displays the estimated probability of a message being fraudulent.
- **Explainable Predictions:** Highlights suspicious words and patterns with simple explanations.
- **Batch Scoring:** Analyzes multiple SMS messages through CSV uploads.

## Features

The application includes six interactive sections:

1. **Data & Quality** – Dataset auditing, cleaning, and quality reports.
2. **Model Performance** – Model comparisons, ROC curves, and confusion matrices.
3. **Single Prediction** – Analyze individual SMS messages.
4. **Batch Scoring** – Upload CSV files and analyze multiple messages.
5. **Settings** – Select models, adjust decision thresholds, and configure cost settings.
6. **Learn the Tricks** – Understand suspicious words and common SMS scam techniques.

## Machine Learning Models

Three classification algorithms are implemented:

| Model | Text Representation |
|---|---|
| Multinomial Naive Bayes | Word Counts |
| Decision Tree | Binary Word Presence |
| K-Nearest Neighbours | TF-IDF |

Models are tuned using GridSearchCV with stratified cross-validation.

## Dataset

The project uses the **UCI SMS Spam Collection Dataset**.

- Original messages: 5,572
- Cleaned messages: 5,157
- Safe (Ham): 4,516
- Spam: 641

## Evaluation

Model performance is evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

The project also considers class imbalance and the higher cost of missing fraudulent messages.

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Visualization:** Matplotlib

## Installation & Setup

**Requirements:** Python 3.10 or newer.

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project directory:

   ```bash
   cd Mid_Lab
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

   Mac/Linux:

   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

The application will open at `https://sms-spam-detector-777.streamlit.app`.

## Project Structure

```text
Mid_Lab/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
├── data/
│   ├── raw/
│   └── clean/
├── docs/
├── models/
└── src/
```

## Academic Information

**Course:** DSC306 – Data Mining  
**Project Title:** SMS Scam and Fraud Shield

---
