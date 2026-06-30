from src.config import ENCODING_MAP, FEATURE_COLS, MODEL_PATHS
from src.utils import load_model
from src.features import build_feature_vector
from joblib import load

# Simple rule-based QA bank (make these match your PDF)
QUESTION_BANK = {
    "sadness": "Have you felt persistently sad recently? (Yes/No/Sometimes)",
    "anxiety": "Have you felt anxious or worried most days? (Yes/No/Sometimes)",
    "irritability": "Have you felt unusually irritable? (Yes/No/Sometimes)",
    "guilt": "Do you feel excessive guilt or worthlessness? (Yes/No/Sometimes)",
    "sleep_issues": "Are you facing trouble sleeping or disturbed sleep? (Yes/No/Sometimes)",
    "bonding_difficulty": "Is bonding with your baby difficult? (Yes/No/Sometimes)",
    "concentration_issues": "Are you having trouble concentrating? (Yes/No/Sometimes)",
    "loss_of_interest": "Have you lost interest in usual activities? (Yes/No/Sometimes)",
    "fatigue": "Do you feel unusually tired or fatigued? (Yes/No/Sometimes)",
    "suicidal_ideation": "Have you had thoughts of harming yourself? (Yes/No/Sometimes)",
    "social_withdrawal": "Are you withdrawing from social interactions? (Yes/No/Sometimes)"
}

def validate_response(resp: str) -> bool:
    return resp.strip().title() in ENCODING_MAP.keys()

def encode_response(resp: str) -> int:
    return ENCODING_MAP[resp.strip().title()]

def get_recommendation(label_str: str, suicidal_encoded: int) -> str:
    if label_str == "Low":
        return ("Low risk detected. Focus on healthy routines, rest, balanced meals, and gentle activity. "
                "If concerns arise, consider speaking with a professional.")
    elif label_str == "Moderate":
        return ("Moderate risk detected. Counseling may be helpful. Reach out to trusted family or peers for support. "
                "Monitoring symptoms and seeking professional guidance is recommended.")
    else:
        if suicidal_encoded == 2:
            return ("High risk detected with concerning thoughts. Please seek immediate in-person support from a qualified professional "
                    "or trusted contacts. You are not alone, and timely help matters.")
        return ("High risk detected. Please contact a qualified mental health professional promptly. "
                "Lean on supportive people around you while you arrange care.")

def run_chatbot():
    print("Welcome to the PPD Risk Screening Assistant.")
    print("Please answer the following questions honestly. Valid responses: Yes / No / Sometimes.\n")

    responses_encoded = {}

    for feature in FEATURE_COLS:
        q = QUESTION_BANK.get(feature, f"Provide input for {feature} (Yes/No/Sometimes)")
        while True:
            user_input = input(q + " ").strip()
            if validate_response(user_input):
                responses_encoded[feature] = encode_response(user_input)
                break
            else:
                print("Invalid input. Please reply with Yes / No / Sometimes.")

    # Build feature vector
    X = build_feature_vector(responses_encoded)

    # Load model + label encoder
    model = load_model(MODEL_PATHS["lightgbm"])
    le = load("models/label_encoder.pkl")

    # Predict probabilities -> class
    probs = model.predict_proba(X)[0]
    label_idx = probs.argmax()
    label_str = le.inverse_transform([label_idx])[0]

    print("\n— Result —")
    print(f"Predicted risk level: {label_str}")
    print(f"Class probabilities (Low/Moderate/High): {probs}")

    rec = get_recommendation(label_str, responses_encoded.get("suicidal_ideation", 0))
    print("\n— Recommendations —")
    print(rec)

if __name__ == "__main__":
    run_chatbot()
