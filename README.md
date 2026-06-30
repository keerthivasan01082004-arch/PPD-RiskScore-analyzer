# 🧠 Rule-Based Chatbot for Postpartum Depression Risk Prediction

An AI-powered postpartum mental health risk assessment system that combines Machine Learning models with a rule-based conversational chatbot to predict postpartum depression and suicide risk levels and provide personalized recommendations.

This system uses Gradient Boosting models (LightGBM, CatBoost, XGBoost) trained on structured postpartum survey data and integrates them into an interactive chatbot interface for real-time screening and guidance.

📄 Conference Paper: Rule-Based Chatbot for Postpartum Depression Risk Prediction  
Status: Presented | Under Publication

---

## 🎯 Objective

To provide an accessible, data-driven, and interpretable digital screening tool for postpartum mental health risk by combining:

- Predictive ML models
- Structured survey encoding
- Risk stratification (Low / Moderate / High)
- Rule-based recommendation engine
- Conversational chatbot interface

---

## 🚀 Key Features

- 🤖 Interactive rule-based chatbot for structured mental health screening
- 📊 ML-based postpartum depression risk prediction
- 🧮 Multi-model training and comparison
- 🏷 Three-level risk classification (Low / Moderate / High)
- 💡 Automated personalized recommendations
- 📈 Model performance evaluation with metrics & confusion matrices
- 🧠 Feature importance visualization
- ⚡ Lightweight and deployable architecture

---

## 🧠 Machine Learning Models Used

- LightGBM — best performing model (selected for deployment)
- CatBoost — categorical feature optimized boosting
- XGBoost — robust gradient boosting baseline

All models trained and evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC metrics

---

## 📓 Google Colab Notebooks (Models)
- LightGBM Training Notebook: **https://colab.research.google.com/drive/1baZqq41OMMBTzKqKIljVVDkkVrJ8lCu1?usp=sharing**
- CatBoost Training Notebook: **https://colab.research.google.com/drive/1tWX6srgicNJt7fuC0xqEhijTnA1r3T_R?usp=sharing**
- XGBoost Training Notebook: **https://colab.research.google.com/drive/1hmHuEBf-EsGKqnEewNRJxiDt3W1TGi2s?usp=sharing**

## 🏗 System Pipeline

User Chatbot Input  
→ Survey Encoding (Yes=2, Sometimes=1, No=0)  
→ Feature Vector Creation  
→ ML Model Prediction  
→ Risk Level Classification  
→ Rule-Based Recommendation Engine  
→ User Guidance Output

---

## 📂 Project Structure


---

## 📊 Dataset

- Structured postpartum survey dataset
- Behavioral + emotional indicators
- Risk factor attributes
- Encoded categorical responses

Dataset used for academic research & model benchmarking.

---

## 💬 Chatbot Logic

- Rule-based question flow
- Structured symptom collection
- Encoded response mapping
- ML inference integration
- Risk-specific guidance messages

Risk Mapping:

- 🔴 High Risk → Urgent clinical consultation
- 🟠 Moderate Risk → Counseling & monitoring
- 🟢 Low Risk → Preventive self-care guidance

---

## 🛠 Tech Stack

Python  
LightGBM  
CatBoost  
XGBoost  
Scikit-learn  
Pandas / NumPy  
Streamlit (if used)  
Rule-based Chatbot Engine  


