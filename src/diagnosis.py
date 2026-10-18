import cv2
import os
import glob
import numpy as np
from detector import detect_image
from descriptors import query_by_image


def get_best_detection(detections):
    if not detections:
        return "none", 0.0

    best = max(
        detections,
        key=lambda detection: detection["confidence"]
    )

    return best["label"], best["confidence"]


def get_cbir_majority_label(
    image_path,
    candidate_paths,
    top_k=5
):
    results = query_by_image(
        image_path,
        candidate_paths,
        top_k=top_k
    )

    labels = []

    print()
    print("CBIR nearest neighbors:")

    for result in results:
        neighbor_path = result["image_path"]

        detections = detect_image(
            neighbor_path
        )

        label, confidence = get_best_detection(
            detections
        )

        print(
            neighbor_path,
            "->",
            label,
            "distance =",
            round(result["distance"], 4)
        )

        if label != "none":
            labels.append(label)

    if not labels:
        return "none"

    majority_label = max(
        set(labels),
        key=labels.count
    )

    return majority_label


def cbir_label_mismatch(
    image_path,
    candidate_paths,
    top_k=5
):
    detections = detect_image(
        image_path
    )

    yolo_label, yolo_confidence = (
        get_best_detection(detections)
    )

    cbir_label = get_cbir_majority_label(
        image_path,
        candidate_paths,
        top_k
    )

    mismatch = (
        yolo_label != cbir_label
        and yolo_label != "none"
        and cbir_label != "none"
    )

    return {
        "yolo_label": yolo_label,
        "yolo_confidence": yolo_confidence,
        "cbir_label": cbir_label,
        "mismatch": mismatch
    }

def blur_stability(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    temp_path = "results/temp_blur.jpg"

    os.makedirs(
        "results",
        exist_ok=True
    )

    cv2.imwrite(
        temp_path,
        blurred
    )

    original_detections = detect_image(
        image_path
    )

    blurred_detections = detect_image(
        temp_path
    )

    original_label, original_confidence = (
        get_best_detection(original_detections)
    )

    blurred_label, blurred_confidence = (
        get_best_detection(blurred_detections)
    )

    changed = (
        original_label != blurred_label
        and original_label != "none"
        and blurred_label != "none"
    )

    return {
        "original_label": original_label,
        "original_confidence": original_confidence,
        "blurred_label": blurred_label,
        "blurred_confidence": blurred_confidence,
        "changed": changed
    }

def high_frequency_energy(image_path):
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    energy = np.mean(
        np.abs(laplacian)
    )

    return float(energy)

if __name__ == "__main__":
    image_path = (
        "data/poisoned/"
        "car1_clock_eps8.jpg"
    )

    candidate_paths = (
        glob.glob("data/clean/*.jpg")
        + glob.glob("data/reference/*.jpg")
    )

    candidate_paths = [
        path
        for path in candidate_paths
        if path != image_path
    ]

    result = cbir_label_mismatch(
        image_path,
        candidate_paths,
        top_k=5
    )

    print()
    print("Diagnosis signal 1:")
    
    print(
        "YOLO label:",
        result["yolo_label"]
    )
    print(
        "YOLO confidence:",
        round(
            result["yolo_confidence"],
            4
        )
    )
    print(
        "CBIR majority label:",
        result["cbir_label"]
    )
    print(
        "CBIR/YOLO mismatch:",
        result["mismatch"]
    )

    print()
    print("Diagnosis signal 2:")

    blur_result = blur_stability(
        image_path
    )

    print(
        "Original label:",
        blur_result["original_label"]
    )

    print(
        "Original confidence:",
        round(
            blur_result["original_confidence"],
            4
        )
    )

    print(
        "Blurred label:",
        blur_result["blurred_label"]
    )

    print(
        "Blurred confidence:",
        round(
            blur_result["blurred_confidence"],
            4
        )
    )

    print(
        "Prediction changed after blur:",
        blur_result["changed"]
    )
    
    
    print()
    print("Diagnosis signal 3:")

    frequency_energy = high_frequency_energy(
        image_path
    )

    print(
        "High-frequency energy:",
        round(frequency_energy, 4)
    )
    print()
    print("High-frequency comparison:")

    clean_path = "data/clean/car1.jpg"

    clean_energy = high_frequency_energy(
        clean_path
    )

    poisoned_energy = high_frequency_energy(
        image_path
    )

    print(
        "Clean image energy:",
        round(clean_energy, 4)
    )

    print(
        "Poisoned image energy:",
        round(poisoned_energy, 4)
    )