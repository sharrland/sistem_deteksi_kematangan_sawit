from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from PIL import Image
import numpy as np
import cv2
import joblib

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte

app = Flask(__name__)

# LOAD MODEL DAN SCALER
model = load_model(
    'best_hybrid_model.h5'
)

scaler = joblib.load(
    'glcm_scaler.pkl'
)

# KONFIGURASI
IMAGE_SIZE = 224
THRESHOLD = 0.75
CLASS_NAMES = [
    'Belum Masak',
    'Masak',
    'Non Sawit',
    'Terlalu Masak'
]

# DESKRIPSI
CLASS_DESCRIPTIONS = {

    'Belum Masak':
    'Buah kelapa sawit masih dalam kondisi mentah dengan warna dominan ungu atau hitam pekat. Kandungan minyak pada buah belum optimal sehingga belum direkomendasikan untuk dipanen.',

    'Masak':
    'Buah kelapa sawit berada pada tingkat kematangan optimal dengan warna merah jingga merata. Kondisi ini menunjukkan buah siap dipanen karena kandungan minyaknya sudah maksimal.',

    'Non Sawit':
    'Gambar yang diunggah tidak teridentifikasi sebagai buah kelapa sawit. Silakan unggah gambar buah kelapa sawit yang jelas agar sistem dapat melakukan deteksi dengan benar.',

    'Terlalu Masak':
    'Buah kelapa sawit terdeteksi dalam kondisi terlalu matang dengan warna cenderung gelap dan banyak brondolan yang mulai lepas. Kondisi ini dapat menurunkan kualitas hasil panen dan kandungan minyak.'
}

# PREPROCESS IMAGE CNN
def preprocess_image(img):

    img = img.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array.astype(
        np.float32
    )

    img_array = preprocess_input(
        img_array
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# EKSTRAKSI GLCM
def extract_glcm_features(img_rgb):

    gray = rgb2gray(img_rgb)

    gray = img_as_ubyte(
        gray
    )

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        'contrast'
    ).mean()

    dissimilarity = graycoprops(
        glcm,
        'dissimilarity'
    ).mean()

    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    ).mean()

    energy = graycoprops(
        glcm,
        'energy'
    ).mean()

    correlation = graycoprops(
        glcm,
        'correlation'
    ).mean()

    asm = graycoprops(
        glcm,
        'ASM'
    ).mean()

    return np.array([
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        asm
    ])


# HALAMAN BERANDA
@app.route('/')
def index():

    return render_template(
        'index.html'
    )


# HALAMAN TENTANG
@app.route('/tentang')
def tentang():

    return render_template(
        'tentang.html'
    )


# HALAMAN DETEKSI
@app.route(
    '/deteksi',
    methods=['GET', 'POST']
)
def deteksi():

    if request.method == 'POST':

        if 'file' not in request.files:

            return jsonify({
                'error':
                'Tidak ada file'
            })

        file = request.files['file']

        if file.filename == '':

            return jsonify({
                'error':
                'File kosong'
            })

        try:

            # LOAD IMAGE
            img = Image.open(
                file.stream
            ).convert('RGB')

            # INPUT CNN
            img_array = preprocess_image(
                img
            )

            # INPUT GLCM
            img_rgb = np.array(
                img
            )

            glcm_feature_original = extract_glcm_features(
                img_rgb
            )

            glcm_feature = scaler.transform(
                [glcm_feature_original]
            )

            # PREDIKSI
            prediction = model.predict(
                [
                    img_array,
                    glcm_feature
                ]
            )

            print("\n====================")
            print("HASIL PREDIKSI")
            print(prediction)
            print("====================\n")

            predicted_class = np.argmax(
                prediction
            )

            confidence = float(
                prediction[0][predicted_class]
            )

            hasil = CLASS_NAMES[
                predicted_class
            ]

            # Threshold opsional
            if confidence < THRESHOLD:

                hasil = 'Non Sawit'

            deskripsi = CLASS_DESCRIPTIONS[
                hasil
            ]

            print(
                "Kelas:",
                hasil
            )

            print(
                "Confidence:",
                confidence
            )

            return jsonify({

                'hasil':
                hasil,

                'confidence':
                round(
                    confidence * 100,
                    2
                ),

                'deskripsi':
                deskripsi,

                'glcm': {

                    'contrast':
                    round(float(glcm_feature_original[0]),4),

                    'dissimilarity':
                    round(float(glcm_feature_original[1]),4),

                    'homogeneity':
                    round(float(glcm_feature_original[2]),4),

                    'energy':
                    round(float(glcm_feature_original[3]),4),

                    'correlation':
                    round(float(glcm_feature_original[4]),4),

                    'asm':
                    round(float(glcm_feature_original[5]),4)

                }

            })

        except Exception as e:

            return jsonify({
                'error':
                str(e)
            })

    return render_template(
        'deteksi.html'
    )


# RUN FLASK
if __name__ == '__main__':

    app.run(
        debug=True
    )