import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from src.preprocess import preprocess_pipeline
from src.utils import load_model
from src.config import MODEL_PATHS

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc_multiclass(y_true_enc, y_prob, title="ROC Curves"):
    y_bin = label_binarize(y_true_enc, classes=[0,1,2])
    plt.figure(figsize=(6,5))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model_key="lightgbm"):
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    from joblib import load
    le = load("models/label_encoder.pkl")
    y_test_enc = le.transform(y_test)

    model = load_model(MODEL_PATHS[model_key])
    y_prob = model.predict_proba(X_test)
    y_pred = y_prob.argmax(axis=1)

    print("Classification report:\n")
    print(classification_report(y_test_enc, y_pred, digits=3))

    plot_confusion(y_test_enc, y_pred, title=f"{model_key} Confusion Matrix")
    plot_roc_multiclass(y_test_enc, y_prob, title=f"{model_key} ROC Curves")

if __name__ == "__main__":
    evaluate_model("lightgbm")
