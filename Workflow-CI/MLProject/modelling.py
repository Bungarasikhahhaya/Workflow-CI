"""
=============================================================================
PROYEK PREDIKSI PENYAKIT JANTUNG (HEART DISEASE PREDICTION)
=============================================================================
Sumber Dataset: UCI Machine Learning Repository / Kaggle
Link Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Gambaran Singkat Dataset:
Dataset ini digunakan untuk memprediksi apakah seorang pasien memiliki risiko 
penyakit jantung berdasarkan 14 atribut klinis kesehatan. Atribut tersebut 
meliputi informasi demografis (usia, jenis kelamin), tipe nyeri dada (cp), 
tekanan darah (trestbps), kolesterol (chol), gula darah (fbs), hasil EKG (restecg), 
detak jantung maksimal (thalach), hingga status akhir (target: 0 = Sehat, 1 = Sakit).
=============================================================================
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_retraining():
    # 1. Integrasi Otomatis ke DagsHub (Mendukung Otomatisasi CI)
    try:
        if "DAGSHUB_TOKEN" in os.environ:
            dagshub.auth.add_app_token(token=os.environ["DAGSHUB_TOKEN"])
            
        dagshub.init(repo_owner='BungaRasikhahhaya', repo_name='Eksperimen_SML_BungaRasikhahHaya', mlflow=True)
        print("✅ Berhasil terhubung ke tracking server DagsHub.")
    except Exception as e:
        print(f"⚠️ Tracking dialihkan ke lokal/default karena: {e}")
    
    # 2. Load Dataset Hasil Preprocessing
    data_dir = "heart_disease_preprocessing" 
    
    if not os.path.exists(data_dir):
        data_dir = "MLProject_Folder/heart_disease_preprocessing"

    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    
    # 3. Aktifkan MLflow Autolog (Menggantikan Manual Logging)
    mlflow.sklearn.autolog()
    
    # 4. Proses Training Model
    with mlflow.start_run(run_name="CI_Automated_Retraining"):
        print("⏳ Sedang melatih ulang model di environment CI dengan MLflow Autolog...")
        
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=2, 
            random_state=42
        )
        
        # Autolog otomatis merekam parameter dan metrik di baris ini
        model.fit(X_train, y_train)
        
        # Evaluasi akhir untuk konfirmasi di terminal
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"✨ Re-training Sukses!")
        print(f"Akurasi Model Baru: {acc:.4f}")
        print("✅ Parameter, metrik, dan artefak model berhasil dicatat otomatis oleh MLflow Autolog.")

if __name__ == "__main__":
    run_retraining()