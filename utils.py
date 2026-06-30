import os
import joblib

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def save_model(model, path):
    if path.endswith(".pkl"):
        joblib.dump(model, path)
    elif path.endswith(".cbm"):
        model.save_model(path)
    else:
        joblib.dump(model, path)

def load_model(path):
    if path.endswith(".pkl"):
        return joblib.load(path)
    elif path.endswith(".cbm"):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(path)
        return model
    else:
        return joblib.load(path)

