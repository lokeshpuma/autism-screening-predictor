# Autism Screening Predictor

A Streamlit app that predicts autism spectrum screening outcomes using a trained **Random Forest** model on AQ-10 questionnaire responses and demographic features.

## Demo

1. Answer the 10 AQ-10 screening questions (Yes/No).
2. Fill in age, gender, ethnicity, and other demographic fields.
3. Click **Get Prediction** to see the screening result and confidence score.

## How it works

| Step | What happens |
| ---- | ------------ |
| 1 | User inputs are cleaned (country/ethnicity/relation mappings match training) |
| 2 | Categorical fields are label-encoded with saved encoders |
| 3 | Random Forest returns ASD / non-ASD class with probability |

## Run locally

```bash
git clone https://github.com/lokeshpuma/movie_recommender.git
cd movie_recommender
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select the repository and set **Main file path** to `app.py`.
4. Click **Deploy** — no API keys required.

## Project structure

```
├── app.py              # Streamlit UI
├── predict.py          # Preprocessing and inference
├── best_model.joblib   # Trained Random Forest model
├── encoders.joblib     # Label encoders for categorical fields
├── train.csv           # Training data (used for result-score lookup)
├── requirements.txt
├── .python-version     # Pins Python 3.12 for Streamlit Cloud
└── .streamlit/
    └── config.toml     # App theme
```

## Tech stack

- Streamlit — web UI
- scikit-learn — Random Forest classifier
- pandas / numpy — data handling

## Disclaimer

This application is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

## License

MIT
