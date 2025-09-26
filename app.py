from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from math import isnan
from flask_cors import CORS
from datetime import datetime, date
from dotenv import load_dotenv
from utils import client

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)

CORS(app)

# Path ke file model dan scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_xgboost.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), "house_prices.csv")

# Load model dan scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[SUCCESS] Model and scaler loaded successfully")
except Exception as e:
    print("[ERROR] Error loading model or scaler:", str(e))
    exit(1)


@app.route("/")
def home():
    return "Welcome to the Home Value Prediction API!"


@app.route("/predict-price", methods=["POST"])
def predict_price():
    try:
        # Ambil input JSON
        data = request.json

        # Validasi input
        required_fields = [
            "kamar_tidur",
            "kamar_mandi",
            "garasi",
            "luas_tanah",
            "luas_bangunan",
            "lokasi",
            "waktu_penjualan",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field {field} wajib diisi"}), 400

        # Ekstrak input
        kamar_tidur = float(data["kamar_tidur"])
        kamar_mandi = float(data["kamar_mandi"])
        garasi = float(data["garasi"])
        luas_tanah = float(data["luas_tanah"])
        luas_bangunan = float(data["luas_bangunan"])
        lokasi = str(data["lokasi"])
        waktu_penjualan = str(data["waktu_penjualan"])

        # Buat DataFrame
        df = pd.DataFrame(
            {
                "kamar_tidur": [kamar_tidur],
                "kamar_mandi": [kamar_mandi],
                "garasi": [garasi],
                "luas_tanah": [luas_tanah],
                "luas_bangunan": [luas_bangunan],
                "lokasi": [lokasi],
                "waktu_penjualan": [waktu_penjualan],
            }
        )

        # One-hot encoding
        df["waktu_penjualan"] = df["waktu_penjualan"].astype(str)
        data_encoded = pd.get_dummies(
            df, columns=["lokasi", "waktu_penjualan"], drop_first=True
        )

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
        return (
            jsonify(
                {
                    "success": True,
                    "prediksi_harga": int(prediksi_harga),
                    "pesan": "Prediksi harga rumah berhasil!",
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/data-analytics", methods=["GET"])
def data_analytics():
    try:
        # Baca file CSV
        data = pd.read_csv(DATA_CSV_PATH)

        # KPI Cards
        total_rumah = len(data)
        harga_minimum = data["harga"].min()
        harga_maksimum = data["harga"].max()
        harga_rata_rata = data["harga"].mean()
        harga_median = data["harga"].median()

        # Price per Square Meter (asumsi kolom 'luas_tanah' digunakan)
        data["harga_per_meter"] = data["harga"] / data["luas_tanah"]
        harga_per_meter_rata = data["harga_per_meter"].mean()

        # Persentase rumah dengan garasi
        total_rumah_dengan_garasi = data[data["garasi"] > 0].shape[0]
        persentase_garasi = (total_rumah_dengan_garasi / total_rumah) * 100

        # Rata-rata kamar tidur & mandi
        avg_kamar_tidur = data["kamar_tidur"].mean()
        avg_kamar_mandi = data["kamar_mandi"].mean()

        # Lokasi termahal dan termurah
        lokasi_termahal = data.loc[data["harga"].idxmax(), "lokasi"]
        harga_lokasi_termahal = data["harga"].max()
        lokasi_termurah = data.loc[data["harga"].idxmin(), "lokasi"]
        harga_lokasi_termurah = data["harga"].min()

        # Grafik 1: Harga Rata-Rata per Lokasi
        grafik_lokasi = (
            data.groupby("lokasi")["harga"]
            .mean()
            .reset_index()
            .sort_values("harga", ascending=False)
            .head(10)  # Ambil 10 lokasi termahal
            .to_dict(orient="records")
        )

        # Grafik 2: Distribusi Harga (Histogram Data)
        grafik_distribusi = data["harga"].tolist()

        # Grafik 4: Proporsi Jumlah Kamar Tidur
        grafik_kamar_tidur = (
            data["kamar_tidur"]
            .value_counts()
            .reset_index()
            .rename(columns={"kamar_tidur": "jumlah_kamar", "count": "jumlah_rumah"})
            .to_dict(orient="records")
        )

        # Tabel: Top 5 Most Expensive Houses
        top_5_mahal = data.nlargest(5, "harga")[
            ["lokasi", "luas_tanah", "luas_bangunan", "harga"]
        ].to_dict(orient="records")

        # Tabel: Top 5 Cheapest Houses
        top_5_murah = data.nsmallest(5, "harga")[
            ["lokasi", "luas_tanah", "luas_bangunan", "harga"]
        ].to_dict(orient="records")

        # Susun data sementara
        response_data = {
            "success": True,
            "kpi_cards": {
                "total_rumah": int(total_rumah),
                "harga_minimum": int(harga_minimum),
                "harga_maksimum": int(harga_maksimum),
                "harga_rata_rata": int(harga_rata_rata),
                "harga_median": int(harga_median),
                "harga_per_meter_rata": int(harga_per_meter_rata),
                "persentase_garasi": round(persentase_garasi, 2),
                "avg_kamar_tidur": round(avg_kamar_tidur, 2),
                "avg_kamar_mandi": round(avg_kamar_mandi, 2),
                "lokasi_termahal": {
                    "lokasi": lokasi_termahal,
                    "harga": int(harga_lokasi_termahal),
                },
                "lokasi_termurah": {
                    "lokasi": lokasi_termurah,
                    "harga": int(harga_lokasi_termurah),
                },
            },
            "grafik": {
                "harga_per_lokasi": grafik_lokasi,
                "distribusi_harga": grafik_distribusi,
                "proporsi_kamar_tidur": grafik_kamar_tidur,
            },
            "tabel": {"top_5_mahal": top_5_mahal, "top_5_murah": top_5_murah},
        }

        # Insight
        prompt = f"""
                Berikut adalah data statistik dan analitik dari dataset properti di Bandung:

                {response_data['kpi_cards']}
                
                dan
                
                {response_data['tabel']}

                Berdasarkan data di atas, buatkan 1-2 kalimat insight menarik dan informatif yang bisa ditampilkan di dashboard analitik website saya. Fokus pada tren harga, lokasi strategis, atau fakta menarik lainnya yang relevan bagi calon pembeli rumah.
                """

        print(f"prompt: {prompt}")

        insight = client.chat.completions.create(
            model="Qwen/Qwen3-Next-80B-A3B-Thinking",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response_data["insight"] = insight.choices[0].message.content
        response_data["message"] = "Data analitik berhasil diambil!"
        response_data["success"] = True

        return (
            jsonify(response_data),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
