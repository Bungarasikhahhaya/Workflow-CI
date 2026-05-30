import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ===== TAMBAHKAN KODE INI UNTUK MENGHUBUNGKAN KE DAGSHUB =====
import dagshub
import mlflow

# Ganti 'username_kamu' dan 'nama_proyek_kamu' sesuai dengan yang ada di URL DagsHub-mu
dagshub.init(repo_owner='username_kamu', repo_name='nama_proyek_kamu', mlflow=True)
mlflow.set_experiment("Heart-Disease-Tuning")
# ============================================================

# 1. Load Dataset Sesuai Struktur Folder
print("Memuat data dari folder heart_disease_preprocessing...")
X_train = pd.read_csv("heart_disease_preprocessing/X_train.csv")
X_test = pd.read_csv("heart_disease_preprocessing/X_test.csv")
y_train = pd.read_csv("heart_disease_preprocessing/y_train.csv")
y_test = pd.read_csv("heart_disease_preprocessing/y_test.csv")

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# 2. Inisialisasi Model Baseline
rf = RandomForestClassifier(random_state=42)

# 3. Tentukan Hyperparameter yang Ingin Diuji (Tuning)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# ===== JALANKAN MLFLOW UNTUK PENCATATAN EKSPERIMEN =====
with mlflow.start_run():
    print("Memulai proses Hyperparameter Tuning dengan GridSearchCV...")
    # 4. Jalankan GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 5. Menampilkan Parameter Terbaik
    best_params = grid_search.best_params_
    print(f"\nKombinasi Parameter Terbaik: {best_params}")

    # Log parameter terbaik ke DagsHub
    mlflow.log_params(best_params)

    # 6. Evaluasi Model Terbaik Hasil Tuning
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n================ EVALUASI MODEL TUNING ================")
    print(f"Akurasi Model Setelah Tuning: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Log metrik akurasi ke DagsHub
    mlflow.log_metric("accuracy", accuracy)

    # 7. Simpan Model Hasil Tuning sebagai Artifak (.pkl) lokal dan cloud
    joblib.dump(best_model, "best_model_heart_disease.pkl")
    mlflow.log_artifact("best_model_heart_disease.pkl")
    print("Model terbaik berhasil disimpan lokal dan dikirim ke DagsHub!")