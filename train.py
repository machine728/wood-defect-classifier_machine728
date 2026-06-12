import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ── Konfigurasi ──────────────────────────────────────────────
DATASET_DIR = "dataset"   # folder berisi subfolder crack / knot / normal
MODEL_PATH  = "model/knn_model.pkl"
# ─────────────────────────────────────────────────────────────

def extract_glcm_features(image_path):
    """Ekstrak 4 fitur GLCM dari satu gambar."""
    img = io.imread(image_path)

    # Konversi ke grayscale
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img

    img_uint8 = (img_gray * 255).astype(np.uint8)

    # Hitung GLCM (jarak=1, sudut=0°)
    glcm = graycomatrix(img_uint8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)

    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return [contrast, energy, homogeneity, correlation]


def load_dataset(dataset_dir):
    """Load semua gambar dari subfolder, return features & labels."""
    features, labels = [], []
    classes = sorted(os.listdir(dataset_dir))

    print(f"Kelas ditemukan: {classes}\n")

    for label in classes:
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"  [{label}] → {len(images)} gambar")

        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            try:
                feat = extract_glcm_features(img_path)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"    Skip {img_file}: {e}")

    return np.array(features), np.array(labels)


def main():
    print("=" * 50)
    print("  GLCM + KNN — Wood Defect Classifier")
    print("=" * 50)

    # 1. Load dataset
    print("\n[1] Loading dataset...")
    X, y = load_dataset(DATASET_DIR)
    print(f"\nTotal sampel: {len(X)}")

    # 2. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[2] Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Train KNN
    print("\n[3] Training KNN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 4. Evaluasi
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[4] Akurasi: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Simpan model
    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": knn, "classes": list(knn.classes_)}, f)
    print(f"\n[5] Model disimpan → {MODEL_PATH}")
    print("\nDone! ✅")


if __name__ == "__main__":
    main()