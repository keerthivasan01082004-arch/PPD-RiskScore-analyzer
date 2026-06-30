import warnings
warnings.filterwarnings("ignore")

from src.preprocess import preprocess_pipeline
from src.config import MODEL_PATHS, RANDOM_STATE, FEATURE_COLS
from src.utils import ensure_dirs, save_model

# Models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

def train_models():
    ensure_dirs()
    X_train, X_test, y_train, y_test = preprocess_pipeline()

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )
    lgbm.fit(X_train, y_train_enc)
    y_pred_lgbm = lgbm.predict(X_test)
    acc_lgbm = accuracy_score(y_test_enc, y_pred_lgbm)
    y_test_bin = label_binarize(y_test_enc, classes=range(len(le.classes_)))
    y_prob_lgbm = lgbm.predict_proba(X_test)
    auc_lgbm = roc_auc_score(y_test_bin, y_prob_lgbm, average="macro")
    print(f"LightGBM -> Accuracy: {acc_lgbm:.3f}, AUC: {auc_lgbm:.3f}")
    print("LightGBM Classification Report:")
    print(classification_report(y_test_enc, y_pred_lgbm))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred_lgbm)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("LightGBM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/lightgbm_confusion_matrix.png")
    plt.close()
    print("LightGBM Confusion Matrix saved as models/lightgbm_confusion_matrix.png")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=RANDOM_STATE
    )
    xgb.fit(X_train, y_train_enc)
    y_prob_xgb = xgb.predict_proba(X_test)
    y_pred_xgb = y_prob_xgb.argmax(axis=1)
    acc_xgb = accuracy_score(y_test_enc, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test_bin, y_prob_xgb, average="macro")
    print(f"XGBoost -> Accuracy: {acc_xgb:.3f}, AUC: {auc_xgb:.3f}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test_enc, y_pred_xgb))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred_xgb)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/xgb_confusion_matrix.png")
    plt.close()
    print("XGBoost Confusion Matrix saved as models/xgb_confusion_matrix.png")

    # CatBoost
    cat = CatBoostClassifier(
        iterations=600,
        depth=4,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False
    )
    cat.fit(X_train, y_train_enc)
    y_prob_cat = cat.predict_proba(X_test)
    y_pred_cat = y_prob_cat.argmax(axis=1)
    acc_cat = accuracy_score(y_test_enc, y_pred_cat)
    auc_cat = roc_auc_score(y_test_bin, y_prob_cat, average="macro")
    print(f"CatBoost -> Accuracy: {acc_cat:.3f}, AUC: {auc_cat:.3f}")
    print("CatBoost Classification Report:")
    print(classification_report(y_test_enc, y_pred_cat))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred_cat)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("CatBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/catboost_confusion_matrix.png")
    plt.close()
    print("CatBoost Confusion Matrix saved as models/catboost_confusion_matrix.png")

    # Save models
    save_model(lgbm, MODEL_PATHS["lightgbm"])
    save_model(xgb, MODEL_PATHS["xgb"])
    save_model(cat, MODEL_PATHS["catboost"])

    # Save label encoder
    joblib.dump(le, "models/label_encoder.pkl")

    # Save metrics
    metrics = {
        "LightGBM": {"accuracy": acc_lgbm, "auc": auc_lgbm},
        "XGBoost": {"accuracy": acc_xgb, "auc": auc_xgb},
        "CatBoost": {"accuracy": acc_cat, "auc": auc_cat}
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Extract and save feature importances
    feature_importances = {
        "LightGBM": dict(zip(FEATURE_COLS, lgbm.feature_importances_.astype(float))),
        "XGBoost": dict(zip(FEATURE_COLS, xgb.feature_importances_.astype(float))),
        "CatBoost": dict(zip(FEATURE_COLS, cat.feature_importances_.astype(float)))
    }
    with open("models/feature_importances.json", "w") as f:
        json.dump(feature_importances, f, indent=4)

    print("✅ Training complete. Models, encoder, and feature importances saved.")

if __name__ == "__main__":
    train_models()
