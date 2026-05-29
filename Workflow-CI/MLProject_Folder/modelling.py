import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_retraining():
    # 1. Integrasi Otomatis ke DagsHub (Mendukung Otomatisasi CI)
    try:
        # Cek apakah token tersedia di environment variable
        if "DAGSHUB_TOKEN" in os.environ:
            dagshub.auth.add_app_token(token=os.environ["DAGSHUB_TOKEN"])
            
        dagshub.init(repo_owner='BungaRasikhahhaya', repo_name='Eksperimen_SML_BungaRasikhahHaya', mlflow=True)
        print("✅ Berhasil terhubung ke tracking server DagsHub.")
    except Exception as e:
        print(f"⚠️ Tracking dialihkan ke lokal/default karena: {e}")
    
    # 2. Load Dataset Hasil Preprocessing
    # Menggunakan path relatif agar fleksibel saat dijalankan di komputer lokal maupun di dalam Docker Container
    data_dir = "heart_disease_preprocessing" 
    
    if not os.path.exists(data_dir):
        # Fallback path jika struktur folder di Docker agak berbeda
        data_dir = "MLProject_Folder/heart_disease_preprocessing"

    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    
    # 3. Proses Training Model
    # Menggunakan parameter tetap (fixed) hasil tuning terbaik kemarin agar eksekusi cepat dan ringan
    with mlflow.start_run(run_name="CI_Automated_Retraining"):
        print("⏳ Sedang melatih ulang model di environment CI...")
        
        # Inisialisasi model dengan hyperparameter terbaik hasil Kriteria 2
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=2, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 4. Evaluasi & Manual Logging (Sesuai Syarat Skilled/Advance)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Catat parameter dan metrik secara manual
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", acc)
        
        # Simpan model sebagai artefak utama yang bisa di-serving nantinya
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✨ Re-training Sukses!")
        print(f"Akurasi Model Baru: {acc:.4f}")
        print("✅ Model baru berhasil disimpan ke MLflow Artifacts.")

if __name__ == "__main__":
    run_retraining()
