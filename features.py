import pandas as pd

def build_feature_vector(responses: dict) -> pd.DataFrame:
    """
    Convert validated chatbot responses (already encoded with 0/1/2) into a single-row DataFrame.
    responses: {feature_name: int}
    """
    return pd.DataFrame([responses])
