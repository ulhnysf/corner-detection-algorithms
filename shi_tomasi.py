import cv2
import os

input_folder = "processed/grayscale"
output_folder = "demo/shi_tomasi"

os.makedirs(output_folder, exist_ok=True)

for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)

    if os.path.isdir(category_path):
        output_category = os.path.join(output_folder, category)
        os.makedirs(output_category, exist_ok=True)

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

            save_path = os.path.join(output_category, filename)
            cv2.imwrite(save_path, img)

print("Shi-Tomasi tamamlandı!")