# 🧠 Autism Screening Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.1-F7931E.svg?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B.svg?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive Machine Learning web application built with **Streamlit** and **Scikit-Learn** that evaluates Autism Spectrum Disorder (ASD) screening probability based on the standardized **AQ-10 (Autism Spectrum Quotient 10-item)** questionnaire and key demographic parameters.

---

## 📌 Overview

Early screening for Autism Spectrum Disorder can assist individuals and healthcare providers in identifying whether a formal clinical assessment is recommended. This repository provides an end-to-end ML pipeline and web interface that:
- Collects response data across 10 behavioral indicators (AQ-10 scale).
- Integrates demographic variables (age, gender, ethnicity, jaundice at birth, family history of autism, country of residence).
- Preprocesses and normalizes categorical inputs using scikit-learn encoders.
- Executes real-time inference using a trained **Random Forest Classifier**.
- Displays binary screening results along with model prediction confidence metrics.

---

## ✨ Features

- 📝 **Standardized AQ-10 Form**: Interactive 10-question behavioral questionnaire with clear Yes/No choices.
- 👤 **Demographic Normalization**: Robust handling of ethnicity, country, and relation features with automated mappings.
- ⚡ **Optimized ML Pipeline**: High-speed inference with pre-loaded model artifacts (`best_model.joblib` and `encoders.joblib`).
- 🎨 **Modern Sleek UI**: Customized dark-mode Streamlit layout with responsive columns and visual probability metrics.
- 🚀 **Streamlit Cloud Ready**: Lightweight dependencies configured for one-click deployment.

---

## 🛠️ Architecture & Pipeline

```
  +-------------------------------------------------------+
  |                   User Inputs                        |
  |  - 10 AQ-10 Questionnaire Responses (0 / 1)           |
  |  - Demographic Data (Age, Gender, Ethnicity, etc.)    |
  +---------------------------+---------------------------+
                              |
                              v
  +-------------------------------------------------------+
  |              Preprocessing Pipeline                   |
  |  - Normalize categorical aliases (e.g. Country/Rel)   |
  |  - Encode features via saved LabelEncoders            |
  |  - Compute result lookup score                        |
  +---------------------------+---------------------------+
                              |
                              v
  +-------------------------------------------------------+
  |             Random Forest Classifier                  |
  |  - Evaluates feature vector against 50 trees          |
  |  - Computes class probability & prediction            |
  +---------------------------+---------------------------+
                              |
                              v
  +-------------------------------------------------------+
  |                   Streamlit Output                    |
  |  - Screening Result (ASD Traits Likely / Not Detected)|
  |  - Model Confidence Percentage                        |
  +-------------------------------------------------------+
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- `git` installed on your machine

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/lokeshpuma/autism-screening-predictor.git
   cd autism-screening-predictor
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (Command Prompt):
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit App**
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This app is optimized for zero-configuration deployment on **Streamlit Community Cloud**:

1. Fork or push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **New app** and configure:
   - **Repository**: `your-username/autism-screening-predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**. Pre-built binary wheels ensure the app installs and launches in under **1 minute**.

---

## 📁 Repository Structure

```
autism-screening-predictor/
├── .streamlit/
│   └── config.toml          # Custom UI theme and server configuration
├── app.py                   # Main Streamlit frontend interface
├── predict.py               # Feature preprocessing and ML inference logic
├── best_model.joblib        # Trained Random Forest classifier artifact
├── encoders.joblib          # Saved scikit-learn LabelEncoders
├── train.csv                # Reference dataset
├── requirements.txt         # Pinned python dependency specifications
├── .python-version          # Python 3.12 runtime specification
└── README.md                # Project documentation
```

---

## 🧰 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend Framework** | [Streamlit](https://streamlit.io/) | Interactive web user interface |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | Random Forest Classifier & LabelEncoders |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) | Matrix manipulation and input formatting |
| **Model Persistence** | [Joblib](https://joblib.readthedocs.io/) | Fast serialization of ML estimators |

---

## ⚠️ Medical Disclaimer

> [!IMPORTANT]
> **This tool is for educational and informational purposes only.** It relies on a machine learning model trained on screening dataset samples and does **not** provide clinical diagnosis or medical assessment. If you have concerns regarding autism spectrum disorder or developmental health, please consult a qualified healthcare professional or clinical specialist.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
