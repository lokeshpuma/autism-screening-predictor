"""Load model artifacts and run autism screening predictions."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.joblib"
ENCODERS_PATH = ROOT / "encoders.joblib"

COUNTRY_MAPPING = {
    "Viet Nam": "Vietnam",
    "AmericanSamoa": "United States",
    "Hong Kong": "China",
}

FEATURE_COLUMNS = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "ethnicity",
    "jaundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "result",
    "relation",
]

AQ_QUESTIONS = {
    "A1_Score": "I often notice small sounds when others do not.",
    "A2_Score": "I usually concentrate more on the whole picture, rather than small details.",
    "A3_Score": "I find it easy to do more than one thing at once.",
    "A4_Score": "If there is an interruption, I can return to what I was doing quickly.",
    "A5_Score": "I find it easy to read between the lines when someone is talking to me.",
    "A6_Score": "I know how to tell if someone listening to me is getting bored.",
    "A7_Score": "When reading a story, I find it difficult to work out characters' intentions.",
    "A8_Score": "I like to collect information about categories of things.",
    "A9_Score": "I find it easy to work out what someone is thinking from their face.",
    "A10_Score": "I find it difficult to work out people's intentions.",
}


def _normalize_categoricals(raw: dict) -> dict:
    data = dict(raw)
    data["ethnicity"] = data["ethnicity"].replace("?", "Others").replace("others", "Others")
    data["contry_of_res"] = COUNTRY_MAPPING.get(data["contry_of_res"], data["contry_of_res"])
    data["relation"] = {
        "?": "Others",
        "Relative": "Others",
        "Parent": "Others",
        "Health care professional": "Others",
    }.get(data["relation"], data["relation"])
    return data


RESULT_LOOKUP: dict[int, float] = {
    0: 6.463854475,
    1: 6.202339451,
    2: 7.095318414,
    3: 8.224135755,
    4: 6.2831959605,
    5: 9.220903954,
    6: 9.91210388,
    7: 11.580762105,
    8: 12.22096627,
    9: 12.32388114,
    10: 13.19584858,
}


def _estimate_result(scores: dict[str, int], lookup: dict[int, float]) -> float:
    score_sum = sum(scores[f"A{i}_Score"] for i in range(1, 11))
    if score_sum in lookup:
        return lookup[score_sum]
    return float(np.mean(list(lookup.values())))


def load_artifacts():
    if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Expected best_model.joblib and encoders.joblib "
            "in the project root."
        )
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders, RESULT_LOOKUP


def encode_features(raw: dict, encoders: dict, lookup: dict[int, float]) -> pd.DataFrame:
    data = _normalize_categoricals(raw)
    scores = {f"A{i}_Score": int(data[f"A{i}_Score"]) for i in range(1, 11)}
    data["result"] = _estimate_result(scores, lookup)

    row = {col: data[col] for col in FEATURE_COLUMNS}
    frame = pd.DataFrame([row])

    for column, encoder in encoders.items():
        frame[column] = encoder.transform(frame[column].astype(str))

    return frame[FEATURE_COLUMNS]


def predict(raw: dict, model, encoders, lookup: dict[int, float]) -> tuple[int, float]:
    features = encode_features(raw, encoders, lookup)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])
    return prediction, probability
