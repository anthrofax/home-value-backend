from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from math import isnan
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Path ke file model dan scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_xgboost.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), 'house_prices.csv')

# Load model dan scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[SUCCESS] Model and scaler loaded successfully")
except Exception as e:
    print("[ERROR] Error loading model or scaler:", str(e))
    exit(1)


@app.route('/')
def home():
    return "Welcome to the Home Value Prediction API!"


@app.route('/predict-price', methods=['POST'])
def predict_price():
    try:
        # Ambil input JSON
        data = request.json

        # Validasi input
        required_fields = ['kamar_tidur', 'kamar_mandi', 'garasi', 'luas_tanah', 'luas_bangunan', 'lokasi', 'waktu_penjualan']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Field {field} wajib diisi'}), 400

        # Ekstrak input
        kamar_tidur = float(data['kamar_tidur'])
        kamar_mandi = float(data['kamar_mandi'])
        garasi = float(data['garasi'])
        luas_tanah = float(data['luas_tanah'])
        luas_bangunan = float(data['luas_bangunan'])
        lokasi = str(data['lokasi'])
        waktu_penjualan = str(data['waktu_penjualan'])

        # Buat DataFrame
        df = pd.DataFrame({
            'kamar_tidur': [kamar_tidur],
            'kamar_mandi': [kamar_mandi],
            'garasi': [garasi],
            'luas_tanah': [luas_tanah],
            'luas_bangunan': [luas_bangunan],
            'lokasi': [lokasi],
            'waktu_penjualan': [waktu_penjualan]
        })

        # One-hot encoding
        df['waktu_penjualan'] = df['waktu_penjualan'].astype(str)
        data_encoded = pd.get_dummies(df, columns=['lokasi', 'waktu_penjualan'], drop_first=True)

        # Tambah kolom yang hilang
        missing_cols = list(set(scaler) - set(data_encoded.columns))
        if missing_cols:
            missing_df = pd.DataFrame(0, index=data_encoded.index, columns=missing_cols)
            data_encoded = pd.concat([data_encoded, missing_df], axis=1)

        # Atur urutan kolom
        data_encoded = data_encoded[scaler]
        
        # Prediksi
        prediksi_harga = model.predict(data_encoded)[0]

        # Kirim hasil
        return jsonify({
            'success': True,
            'prediksi_harga': int(prediksi_harga),
            'pesan': 'Prediksi harga rumah berhasil!'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/data-analytics', methods=['GET'])
def data_analytics():
    try:
        # Baca file CSV
        prices = []
        with open(DATA_CSV_PATH, 'r') as file:
            reader = pd.read_csv(file)
            for _, row in reader.iterrows():
                harga = row['harga']
                if not isnan(harga):
                    prices.append(harga)

        # Hitung statistik
        total_rumah = len(prices)
        harga_minimum = min(prices)
        harga_maksimum = max(prices)
        harga_rata_rata = sum(prices) / total_rumah

        # Kirim hasil
        return jsonify({
            'success': True,
            'statistik': {
                'total_rumah': total_rumah,
                'harga_minimum': int(harga_minimum),
                'harga_maksimum': int(harga_maksimum),
                'harga_rata_rata': int(harga_rata_rata)
            },
            'pesan': 'Statistik harga rumah berhasil dihitung!'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)