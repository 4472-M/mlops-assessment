# test_model.py

import pickle
import pandas as pd

def test_model_prediction():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("test_data.csv")
    X = df.drop("target", axis=1)
    preds = model.predict(X)

    assert len(preds) == len(X)
