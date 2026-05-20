from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model CNN
model = load_model('model_sawit.h5')

# Ukuran gambar
IMAGE_SIZE = 224

# Threshold confidence minimal
THRESHOLD = 0.75

# Nama kelas
CLASS_NAMES = [
    'Belum Masak',
    'Masak',
    'Terlalu Masak'
]

# Deskripsi kelas
CLASS_DESCRIPTIONS = {

    'Belum Masak':
    'Buah sawit masih mentah dan belum siap panen.',

    'Masak':
    'Buah sawit matang sempurna dan siap dipanen.',

    'Terlalu Masak':
    'Buah sawit terlalu matang dan mulai membusuk.'
}

# Preprocessing gambar
def preprocess_image(img):

    # Resize gambar
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Konversi ke array
    img_array = np.array(img)

    # Jika gambar memiliki alpha channel (RGBA)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Normalisasi
    img_array = img_array / 255.0

    # Menambah dimensi batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Halaman beranda
@app.route('/')
def index():
    return render_template('index.html')

# Halaman tentang
@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

# Halaman deteksi
@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():

    if request.method == 'POST':

        # Cek apakah file ada
        if 'file' not in request.files:
            return jsonify({
                'error': 'Tidak ada file'
            })

        file = request.files['file']

        # Cek file kosong
        if file.filename == '':
            return jsonify({
                'error': 'File kosong'
            })

        try:

            # Membaca gambar
            img = Image.open(file.stream).convert('RGB')

            # Preprocessing gambar
            img_array = preprocess_image(img)

            # Prediksi model
            prediction = model.predict(img_array)

            # Ambil index kelas dengan nilai tertinggi
            predicted_class = np.argmax(prediction)

            # Ambil confidence tertinggi
            confidence = float(prediction[0][predicted_class])

            # Jika confidence rendah
            if confidence < THRESHOLD:
                return jsonify({
                    'hasil': 'Bukan Buah Sawit',
                    'confidence': round(confidence * 100, 2),
                    'deskripsi': 'Gambar yang diunggah bukan buah kelapa sawit.'
                })

            # Jika confidence memenuhi threshold
            hasil = CLASS_NAMES[predicted_class]

            # Ambil deskripsi
            deskripsi = CLASS_DESCRIPTIONS[hasil]

            # Kirim hasil ke frontend
            return jsonify({
                'hasil': hasil,
                'confidence': round(confidence * 100, 2),
                'deskripsi': deskripsi
            })

        except Exception as e:
            return jsonify({
                'error': str(e)
            })

    return render_template('deteksi.html')

# Menjalankan Flask
if __name__ == '__main__':
    app.run(debug=True)