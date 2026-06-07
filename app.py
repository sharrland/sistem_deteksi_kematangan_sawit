from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
from PIL import Image
import numpy as np
import cv2
import joblib

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte

app = Flask(__name__)

# LOAD MODEL DAN SCALER
MODEL_PATH = "mobilenetv2_glcm_hybrid.h5"
SCALER_PATH = "glcm_scaler.pkl"

try:
    model = load_model(MODEL_PATH)
    print("Model berhasil dimuat")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    raise

try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler berhasil dimuat")
except Exception as e:
    print(f"Gagal memuat scaler: {e}")
    raise

# KONFIGURASI
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Belum Masak",
    "Masak",
    "Non Sawit",
    "Terlalu Masak"
]

# DESKRIPSI KELAS
CLASS_DESCRIPTIONS = {

    "Belum Masak":
    "Buah kelapa sawit masih dalam kondisi mentah dengan warna dominan ungu atau hitam pekat. Kandungan minyak pada buah belum optimal sehingga belum direkomendasikan untuk dipanen.",

    "Masak":
    "Buah kelapa sawit berada pada tingkat kematangan optimal dengan warna merah jingga merata. Kondisi ini menunjukkan buah siap dipanen karena kandungan minyaknya sudah maksimal.",

    "Non Sawit":
    "Gambar yang diunggah tidak teridentifikasi sebagai buah kelapa sawit. Silakan unggah gambar buah kelapa sawit yang jelas agar sistem dapat melakukan deteksi dengan benar.",

    "Terlalu Masak":
    "Buah kelapa sawit terdeteksi dalam kondisi terlalu matang dengan warna cenderung gelap dan banyak brondolan yang mulai lepas. Kondisi ini dapat menurunkan kualitas hasil panen dan kandungan minyak."
}

# PREPROCESS CNN
def preprocess_image(img):

    img = img.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack(
            (img_array,) * 3,
            axis=-1
        )

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
        "contrast"
    ).mean()

    dissimilarity = graycoprops(
        glcm,
        "dissimilarity"
    ).mean()

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    ).mean()

    energy = graycoprops(
        glcm,
        "energy"
    ).mean()

    correlation = graycoprops(
        glcm,
        "correlation"
    ).mean()

    asm = graycoprops(
        glcm,
        "ASM"
    ).mean()

    return np.array([
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        asm
    ])

# HALAMAN UTAMA
@app.route("/")
def index():

    return render_template(
        "index.html"
    )

# HALAMAN TENTANG
@app.route("/tentang")
def tentang():

    return render_template(
        "tentang.html"
    )

# HALAMAN DETEKSI
@app.route(
    "/deteksi",
    methods=["GET", "POST"]
)
def deteksi():

    if request.method == "POST":

        if "file" not in request.files:
            return jsonify({
                "error":
                "Tidak ada file yang diunggah"
            })

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error":
                "File kosong"
            })

        try:

            # LOAD GAMBAR
            img = Image.open(
                file.stream
            ).convert("RGB")

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
                ],
                verbose=0
            )

            predicted_class = int(
                np.argmax(prediction)
            )

            confidence = float(
                prediction[0][predicted_class]
            )

            hasil = CLASS_NAMES[
                predicted_class
            ]

            deskripsi = CLASS_DESCRIPTIONS[
                hasil
            ]

            print("\n========== HASIL ==========")
            print("Probabilitas :", prediction)
            print("Kelas        :", hasil)
            print("Confidence   :", confidence)
            print("===========================\n")

            return jsonify({

                "hasil":
                hasil,

                "confidence":
                round(
                    confidence * 100,
                    2
                ),

                "deskripsi":
                deskripsi,

                "glcm": {

                    "contrast":
                    round(
                        float(glcm_feature_original[0]),
                        4
                    ),

                    "dissimilarity":
                    round(
                        float(glcm_feature_original[1]),
                        4
                    ),

                    "homogeneity":
                    round(
                        float(glcm_feature_original[2]),
                        4
                    ),

                    "energy":
                    round(
                        float(glcm_feature_original[3]),
                        4
                    ),

                    "correlation":
                    round(
                        float(glcm_feature_original[4]),
                        4
                    ),

                    "asm":
                    round(
                        float(glcm_feature_original[5]),
                        4
                    )
                }
            })

        except Exception as e:

            print("ERROR :", str(e))

            return jsonify({
                "error":
                str(e)
            })

    return render_template(
        "deteksi.html"
    )

# RUN FLASK
if __name__ == "__main__":

    app.run(
        debug=True
    )