"""Tekrarlanabilir, paylasilabilir 15 goruntuluk kose/anahtar nokta benchmark'i.

Orijinal all_metrics.py, all_metrics_results.csv ve all_metrics_summary.csv
bu betik tarafindan degistirilmez. Yerel texture goruntuleri dahil edilmez.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CATEGORIES = ("building", "indoor", "road")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def measure_harris(gray: np.ndarray) -> int:
    response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    response = cv2.dilate(response, None)
    return int(np.count_nonzero(response > 0.01 * response.max()))


def measure_shi_tomasi(gray: np.ndarray) -> int:
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    return 0 if corners is None else len(corners)


def main() -> None:
    parser = argparse.ArgumentParser(description="15 paylasilabilir goruntude tekrarlı benchmark")
    parser.add_argument("--repeats", type=int, default=20, help="Goruntu basina olcum sayisi (varsayilan: 20)")
    parser.add_argument("--warmups", type=int, default=3, help="Zamanlanmayan isinma sayisi (varsayilan: 3)")
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "processed" / "grayscale")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args()

    if args.repeats < 2 or args.warmups < 0:
        parser.error("--repeats en az 2, --warmups en az 0 olmali.")

    repeat_csv = args.output_dir / "benchmark_v2_repeats.csv"
    summary_csv = args.output_dir / "benchmark_v2_summary.csv"
    for path in (repeat_csv, summary_csv):
        if path.exists():
            parser.error(f"Var olan dosya korunuyor: {path} (ustune yazilmayacak)")

    # Sabit kategoriler sayesinde lisans durumu belirsiz 'texture' gorselleri
    # ve onlarin yerel islenmis kopyalari test kapsaminda yer almaz.
    image_files: list[tuple[str, Path]] = []
    for category in CATEGORIES:
        category_dir = args.input_dir / category
        if not category_dir.is_dir():
            parser.error(f"Kategori klasoru bulunamadi: {category_dir}")
        files = sorted(p for p in category_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        print(f"{category}: {len(files)} goruntu")
        image_files.extend((category, p) for p in files)
    if not image_files:
        parser.error("Test icin goruntu bulunamadi.")

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setRNGSeed(42)

    fast_detector = cv2.FastFeatureDetector_create()
    orb_detector = cv2.ORB_create(nfeatures=500)

    def measure_fast(gray: np.ndarray) -> int:
        return len(fast_detector.detect(gray, None))

    def measure_orb(gray: np.ndarray) -> int:
        keypoints, _descriptors = orb_detector.detectAndCompute(gray, None)
        return len(keypoints)

    algorithms = (
        ("Harris", "Esik_ustu_piksel", measure_harris),
        ("Shi-Tomasi", "Kose_noktasi", measure_shi_tomasi),
        ("FAST", "Anahtar_nokta", measure_fast),
        ("ORB", "Anahtar_nokta", measure_orb),
    )

    rows: list[dict[str, object]] = []
    print(f"\nToplam {len(image_files)} goruntu; {args.warmups} isinma + {args.repeats} olcum/algoritma")
    print("OpenCV:", cv2.__version__, "| Python:", sys.version.split()[0])
    print("Texture kategorisi kapsam disinda. Goruntu okuma islemi sureye dahil degil.\n")

    for category, path in image_files:
        # np.fromfile + imdecode: Windows'ta Turkce karakterli dizinleri destekler.
        gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Goruntu okunamadi: {path}")
        for name, unit, detector in algorithms:
            for _ in range(args.warmups):
                detector(gray)
            for repetition in range(1, args.repeats + 1):
                start_ns = time.perf_counter_ns()
                count = detector(gray)
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                rows.append({
                    "Kategori": category,
                    "Dosya": path.name,
                    "Algoritma": name,
                    "Tekrar": repetition,
                    "Tespit_Birimi": unit,
                    "Tespit_Sayisi": count,
                    "Sure_ms": elapsed_ms,
                })
        print(f"Islendi: {category}/{path.name}")

    raw = pd.DataFrame(rows)
    # Her goruntunun tekrarlardan gelen ortalamasi hesaplanir; boylece her
    # goruntu algoritma ozetinde esit agirliga sahiptir.
    per_image = raw.groupby(["Kategori", "Dosya", "Algoritma"], as_index=False).agg(
        Ortalama_Tespit_Sayisi=("Tespit_Sayisi", "mean"),
        Ortalama_Sure_ms=("Sure_ms", "mean"),
        Tekrar_Std_ms=("Sure_ms", "std"),
    )
    summary = per_image.groupby("Algoritma", as_index=False).agg(
        Goruntu_Sayisi=("Dosya", "size"),
        Ortalama_Tespit_Sayisi=("Ortalama_Tespit_Sayisi", "mean"),
        Ortalama_Sure_ms=("Ortalama_Sure_ms", "mean"),
        Medyan_Sure_ms=("Ortalama_Sure_ms", "median"),
        Goruntuler_Arasi_Std_ms=("Ortalama_Sure_ms", "std"),
        Ortalama_Tekrar_Std_ms=("Tekrar_Std_ms", "mean"),
    )
    summary["Tekrar_Sayisi"] = args.repeats

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(repeat_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.4f")

    print("\n=== BENCHMARK OZETI ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nYeni dosyalar:")
    print("-", repeat_csv)
    print("-", summary_csv)
    print("\nNOT: Harris esik-ustu piksel sayar; ORB descriptor hesaplar.")
    print("Sure ve tespit sayilari dogruluk (accuracy) gostergesi degildir.")


if __name__ == "__main__":
    main()
