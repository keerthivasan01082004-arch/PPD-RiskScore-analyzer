# Core configuration for columns, encoding, and paths.
DATA_PATH = r"C:\Users\PRATHIBA\Desktop\ppd\data\post natal data.csv"

# Target column for suicide attempt prediction
TARGET_COL = "Suicide attempt"

# Example feature columns (replace with your exact 11 keys from the PDF)
FEATURE_COLS = ['Timestamp', 'Age', 'Feeling sad or Tearful',
       'Irritable towards baby & partner', 'Trouble sleeping at night',
       'Problems concentrating or making decision',
       'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt',
       'Problems of bonding with baby']


# Columns to drop from raw data (timestamps, ids, etc.)
DROP_COLS = ["timestamp"]

# Encoding map for categorical survey responses
ENCODING_MAP = {
    "Yes": 2,
    "Sometimes": 1,
    "No": 0
}

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model save paths
MODEL_PATHS = {
    "lightgbm": "models/lightgbm_ppd.pkl",
    "xgb": "models/xgb_ppd.pkl",
    "catboost": "models/catboost_ppd.cbm"
}
