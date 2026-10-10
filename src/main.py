import csv
import os

from attack import create_targeted_perturbation
from detector import detect_image, evaluate_attack_success
from metrics import calculate_psnr, calculate_ssim


EPSILONS = [2, 4, 6, 8]
TARGET_LABEL = "clock"
CONFIDENCE_THRESHOLD = 0.60


def get_best_detection(detections):
    if not detections:
        return "none", 0.0

    best_detection = max(
        detections,
        key=lambda detection: detection["confidence"]
    )

    return (
        best_detection["label"],
        best_detection["confidence"]
    )


def run_experiments():
    os.makedirs("data/poisoned", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    csv_path = "experiments/experiments.csv"

    fieldnames = [
        "image",
        "attack",
        "target",
        "epsilon",
        "original_label",
        "original_confidence",
        "poisoned_label",
        "poisoned_confidence",
        "success",
        "psnr",
        "ssim"
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
    csv_file,
    fieldnames=fieldnames,
    lineterminator="\n"
)

        writer.writeheader()

        for image_number in range(1, 11):
            clean_path = (
                f"data/clean/car{image_number}.jpg"
            )

            if not os.path.exists(clean_path):
                print("Skipping missing image:", clean_path)
                continue

            original_detections = detect_image(clean_path)

            original_label, original_confidence = (
                get_best_detection(original_detections)
            )

            for epsilon in EPSILONS:
                poisoned_path = (
                    f"data/poisoned/"
                    f"car{image_number}_clock_eps{epsilon}.jpg"
                )

                create_targeted_perturbation(
                    clean_path,
                    "data/reference/clock1.jpg",
                    poisoned_path,
                    epsilon=epsilon
                )

                poisoned_detections = detect_image(
                    poisoned_path
                )

                poisoned_label, poisoned_confidence = (
                    get_best_detection(poisoned_detections)
                )

                success, target_confidence = (
                    evaluate_attack_success(
                        poisoned_detections,
                        target_label=TARGET_LABEL,
                        confidence_threshold=(
                            CONFIDENCE_THRESHOLD
                        )
                    )
                )

                psnr = calculate_psnr(
                    clean_path,
                    poisoned_path
                )

                ssim = calculate_ssim(
                    clean_path,
                    poisoned_path
                )

                writer.writerow({
                    "image": f"car{image_number}.jpg",
                    "attack": (
                        "target_guided_bounded_perturbation"
                    ),
                    "target": TARGET_LABEL,
                    "epsilon": epsilon,
                    "original_label": original_label,
                    "original_confidence": round(
                        original_confidence,
                        4
                    ),
                    "poisoned_label": poisoned_label,
                    "poisoned_confidence": round(
                        poisoned_confidence,
                        4
                    ),
                    "success": success,
                    "psnr": round(psnr, 4),
                    "ssim": round(ssim, 4)
                })

                print(
                    f"car{image_number}.jpg | "
                    f"epsilon={epsilon} | "
                    f"{original_label} -> "
                    f"{poisoned_label} | "
                    f"target confidence="
                    f"{target_confidence:.4f} | "
                    f"success={success}"
                )

    print()
    print("Experiments saved to:", csv_path)


if __name__ == "__main__":
    run_experiments()