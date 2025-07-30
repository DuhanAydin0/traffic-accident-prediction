import pandas as pd



df = pd.read_csv("/Users/duhanaydin/DataBoss/DataBoss3/data/weather_all_cleaned.csv")  # gerekirse path'i değiştir

df["is_saganak"] = (df["precipitation"] > 5).astype(int)

# şehir bazında toplam saat sayısı ve sağanak saat sayısı
summary = df.groupby("city").agg(
    total_hours=("is_saganak", "count"),
    saganak_hours=("is_saganak", "sum")
)

# yüzdelik oran
summary["saganak_oran"] = (summary["saganak_hours"] / summary["total_hours"]) * 100

# en çoktan aza sırala
summary = summary.sort_values("saganak_oran", ascending=False)

summary.to_csv("saganak_oran_by_city.csv")
print("✅ Şehir bazlı sağanak oranı dosyaya yazıldı: saganak_oran_by_city.csv")
