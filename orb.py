import cv2
import os
import csv
import time

INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "demo/orb"
METRICS_PATH = "results/orb_metrics.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("results", exist_ok=True)

orb = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20
)

rows = []
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for category in sorted(os.listdir(INPUT_FOLDER)):
    category_path = os.path.join(INPUT_FOLDER, category)

    if not os.path.isdir(category_path):
        continue

    output_category = os.path.join(OUTPUT_FOLDER, category)
    os.makedirs(output_category, exist_ok=True)

    for filename in sorted(os.listdir(category_path)):
        if not filename.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(category_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Okunamadı: {img_path}")
            continue

        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        start = time.perf_counter()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        elapsed_ms = (time.perf_counter() - start) * 1000

        output_img = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )

        name, ext = os.path.splitext(filename)
        save_name = f"orb_{category}_{name}.jpg"
        save_path = os.path.join(output_category, save_name)

        cv2.imwrite(save_path, output_img)

        rows.append({
            "category": category,
            "filename": filename,
            "algorithm": "ORB",
            "keypoint_count": len(keypoints),
            "time_ms": round(elapsed_ms, 4),
            "descriptor_shape": str(descriptors.shape) if descriptors is not None else "None"
        })

with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "filename",
            "algorithm",
            "keypoint_count",
            "time_ms",
            "descriptor_shape"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"ORB tamamlandı. {len(rows)} görüntü işlendi.")
print(f"Sonuç klasörü: {OUTPUT_FOLDER}")
print(f"Performans tablosu: {METRICS_PATH}")