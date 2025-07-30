import streamlit as st
import joblib
import pandas as pd

model = joblib.load("/Users/duhanaydin/DataBoss/DataBoss3/models_op2/xgboost_op2.pkl")
encoder = joblib.load("/Users/duhanaydin/DataBoss/DataBoss3/models_op2/encoder_op2.pkl")
scaler = joblib.load("/Users/duhanaydin/DataBoss/DataBoss3/models_op2/scaler_op2.pkl")
feature_order = joblib.load("/Users/duhanaydin/DataBoss/DataBoss3/models_op2/feature_order_op2.pkl")

# ------------------
# Kullanıcı Arayüzü
# ------------------
st.title(" Saatlik Kaza Tahmin Sistemi")

# Dinamik şehir ve yağış türü listesi
city = st.selectbox(" Şehir", encoder.categories_[0])

hour = st.slider(" Saat (0–23)", min_value=0, max_value=23, value=12)
temperature = st.number_input(" Sıcaklık (°C)", value=20.0, step=0.5)
windspeed = st.number_input(" Rüzgar Hızı (km/saat)", value=10.0, step=0.5)
precipitation = st.number_input("️ Yağış Miktarı (mm)", value=0.0, step=0.1)


## yağış türü kontrolü, eğer kullanıcı mm = 0 seçerse, yağış türü yok olarak belirliyoruz modelin kafası karışıyor
if precipitation == 0:
    rain_type = "yok"
    st.info("Yağış miktarı 0 mm olduğu için yağış türü 'yok' olarak ayarlandı.")
else:
    rain_type = st.selectbox(" Yağış Türü", encoder.categories_[1])

# ------------------
# Feature Engineering
# ------------------
is_night = int(hour in [0,1,2,3,4,23])
dangerous_temp = int((temperature < 0) or (temperature > 35))
high_wind = int(windspeed > 20)

# risky_rain (train_op2.py’ye göre)
if precipitation > 0 and precipitation <= 2:
    risky_rain = 1.0
elif precipitation > 2 and precipitation <= 5:
    risky_rain = 0.5
else:
    risky_rain = 0.0

# is_saganak (augmented_mm.py’ye göre threshold > 5 mm)
is_saganak = int(precipitation > 5)

# ------------------
# Veri Hazırlığı
# ------------------
input_dict = {
    "hour": hour,
    "temperature_2m": temperature,
    "windspeed_10m": windspeed,
    "precipitation": precipitation,
    "is_night": is_night,
    "dangerous_temp": dangerous_temp,
    "high_wind": high_wind,
    "risky_rain": risky_rain,
    "city": city,
    "yağış_türü": rain_type,
    "is_saganak": is_saganak
}

input_df = pd.DataFrame([input_dict])

# Encode kategorikler
encoded = encoder.transform(input_df[["city", "yağış_türü"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["city", "yağış_türü"]))

# Scale sayısal veriler (is_saganak hariç)
num_cols = ["hour", "temperature_2m", "windspeed_10m", "precipitation",
            "is_night", "dangerous_temp", "high_wind", "risky_rain"]

scaled = scaler.transform(input_df[num_cols])
scaled_df = pd.DataFrame(scaled, columns=num_cols)

# is_saganak'ı normallemeden ekle
scaled_df["is_saganak"] = is_saganak

# Birleştir
final_input = pd.concat([scaled_df, encoded_df], axis=1)
final_input = final_input.reindex(columns=feature_order, fill_value=0)

# ------------------
# Tahmin
# ------------------
if st.button("🔍 Tahmini Gör"):
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]

    st.markdown(f"###  Saat: {hour} -  Şehir: {city}")
    if pred == 1:
        st.error(f"⚠ 1 saat içinde **kaza olabilir!** (Olasılık: %{prob*100:.1f})")
    else:
        st.success(f" Kaza beklenmiyor. (Olasılık: %{prob*100:.1f})")