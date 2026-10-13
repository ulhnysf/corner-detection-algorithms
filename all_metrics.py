import cv2
import os
import time
import numpy as np
import pandas as pd

input_folder = "processed/grayscale"
output_csv = "all_metrics_results.csv"

results = []

for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)

    if not os.path.isdir(category_path):
        continue

    for filename in os.listdir(category_path):
        img_path = os.path.join(category_path, filename)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HARRIS
        start = time.time()
        harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        harris = cv2.dilate(harris, None)
        harris_count = np.sum(harris > 0.01 * harris.max())
        harris_time = (time.time() - start) * 1000

        results.append([category, filename, "Harris", int(harris_count), round(harris_time, 4)])

        # SHI-TOMASI
        start = time.time()
        shi_corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        shi_count = len(shi_corners) if shi_corners is not None else 0
        shi_time = (time.time() - start) * 1000

        results.append([category, filename, "Shi-Tomasi", int(shi_count), round(shi_time, 4)])

        # FAST
        fast = cv2.FastFeatureDetector_create()
        start = time.time()
        fast_keypoints = fast.detect(gray, None)
        fast_count = len(fast_keypoints)
        fast_time = (time.time() - start) * 1000

        results.append([category, filename, "FAST", int(fast_count), round(fast_time, 4)])

        # ORB
        orb = cv2.ORB_create(nfeatures=500)
        start = time.time()
        orb_keypoints, descriptors = orb.detectAndCompute(gray, None)
        orb_count = len(orb_keypoints) if orb_keypoints is not None else 0
        orb_time = (time.time() - start) * 1000

        results.append([category, filename, "ORB", int(orb_count), round(orb_time, 4)])

df = pd.DataFrame(results, columns=[
    "Kategori",
    "Dosya",
    "Algoritma",
    "Kose_Sayisi",
    "Sure_ms"
])

df.to_csv(output_csv, index=False, encoding="utf-8-sig")

summary = df.groupby("Algoritma").agg({
    "Kose_Sayisi": "mean",
    "Sure_ms": "mean"
}).reset_index()

summary["Kose_Sayisi"] = summary["Kose_Sayisi"].round(2)
summary["Sure_ms"] = summary["Sure_ms"].round(4)

print("\n=== TUM GORUNTULER ICIN ORTALAMA METRIK SONUCLARI ===\n")
print(summary.to_string(index=False))

summary.to_csv("all_metrics_summary.csv", index=False, encoding="utf-8-sig")

print("\nDosyalar olusturuldu:")
print("- all_metrics_results.csv")
print("- all_metrics_summary.csv")