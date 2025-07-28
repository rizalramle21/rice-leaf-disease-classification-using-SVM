import pandas as pd
import numpy as np
import cv2
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Konstanta
DATA_FILE = 'klasifikasi_daun (35).csv'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Mengecek apakah ekstensi file gambar diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mengecek objek benar" daun
def is_leaf_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_leaf = np.array([20, 30, 30])   
    upper_leaf = np.array([90, 255, 255])   
    mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    ratio_leaf_pixels = np.count_nonzero(mask) / (image.shape[0] * image.shape[1])

    return ratio_leaf_pixels > 0.10

# Ekstraksi fitur GLCM + HSV
def extract_features(image):
    # Resize 
    resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Ekstraksi GLCM
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Ekstraksi HSV
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

    h, s, v = cv2.split(masked_hsv)
    h_mean = np.mean(h[mask > 0]) if np.any(mask > 0) else 0
    s_mean = np.mean(s[mask > 0]) if np.any(mask > 0) else 0
    v_mean = np.mean(v[mask > 0]) if np.any(mask > 0) else 0

    # Gabungkan fitur
    feature_vector = np.array([contrast, homogeneity, energy, correlation, h_mean, s_mean, v_mean])
    return feature_vector

# Membuat model SVM
def create_svm_model(data_file):
    data = pd.read_csv(data_file)
    X = data[['Contrast', 'Homogeneity', 'Energy', 'Correlation', 'Hue_Mean', 'Saturation_Mean', 'Value_Mean']]
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = make_pipeline(
        StandardScaler(),
        SVC(C=100, gamma=0.1)
    )

    model.fit(X_train, y_train)
    model.fit(X, y)  # retrain full data (opsional)
    return model

# Konversi hasil prediksi menjadi label teks
def get_klasifikasi_keterangan(hasil_klasifikasi):
    if hasil_klasifikasi == 1:
        return "Leaf Blight"
    elif hasil_klasifikasi == 2:
        return "Leaf Smut"
    elif hasil_klasifikasi == 3:
        return "Brown Spot"
    elif hasil_klasifikasi == 4:
        return "Hispa"
    elif hasil_klasifikasi == 5:
        return "Blast"
    else:
        return "Upload Gambar Terlebih Dahulu"

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Halaman klasifikasi
@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    hasil_klasifikasi = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('klasifikasi.html', hasil_klasifikasi="Upload File Terlebih Dahulu")

        image = request.files['image']
        if image.filename == '' or not allowed_file(image.filename):
            return render_template('klasifikasi.html', hasil_klasifikasi="Pilih file gambar dengan ekstensi JPEG atau JPG atau PNG")

        # Konversi file ke array OpenCV
        image_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

        if image_array is None:
            return render_template('klasifikasi.html', hasil_klasifikasi="Gagal membaca gambar. Pastikan file valid.")

        # Cek apakah gambar kemungkinan besar merupakan daun
        if not is_leaf_image(image_array):
            return render_template('klasifikasi.html', hasil_klasifikasi="Gambar tidak terdeteksi sebagai daun.")
        
        # Ekstrak fitur dan prediksi
        new_data = extract_features(image_array).reshape(1, -1)
        model = create_svm_model(DATA_FILE)
        hasil_klasifikasi = model.predict(new_data)[0]

    keterangan_klasifikasi = get_klasifikasi_keterangan(hasil_klasifikasi)
    return render_template('klasifikasi.html', hasil_klasifikasi=keterangan_klasifikasi)

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
