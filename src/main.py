import csv
import os
import shutil

from attack import create_targeted_perturbation
from detector import detect_image, evaluate_attack_success
from metrics import calculate_psnr, calculate_ssim
from pathlib import Path
from uuid import uuid4
from database import insert_ingested_image
from descriptors import extract_histogram
from diagnosis import cbir_label_mismatch, blur_stability
from filters import restore_image
from database import store_restoration
from database import search_images_by_label
from attack_pgd import targeted_pgd
from database import store_attack
from database import (
    create_tables,
    list_images,
    get_image_lineage,
    get_audit_logs,
)

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

def ingest_image():
    entered_path = input("Enter image path: ").strip().strip("\"'")
    source = Path(entered_path).expanduser()

    if not source.is_file():
        print("Image file not found:", source)
        return

    destination = None

    try:
        descriptor = extract_histogram(str(source))
        detections = detect_image(str(source))

        folder = Path("data/ingested")
        folder.mkdir(parents=True, exist_ok=True)

        destination = folder / (
            f"{uuid4().hex}{source.suffix.lower()}"
        )
        shutil.copy2(source, destination)

        image_id = insert_ingested_image(
            str(destination),
            detections,
            descriptor
        )

    except Exception as error:
        if destination is not None:
            destination.unlink(missing_ok=True)

        print("Ingest failed:", error)
        return

    label, confidence = get_best_detection(detections)

    print("\nImage stored successfully.")
    print("Database ID:", image_id)
    print("Stored file:", destination)
    print("Top label:", label)
    print("Confidence:", round(confidence, 4))
    print("Descriptor length:", len(descriptor))
    print("Status: unverified")

def search_images():
    label = input("Enter object label: ").strip()

    if not label:
        print("Please enter a label.")
        return

    results = search_images_by_label(label)

    if not results:
        print("No matching images in the database.")
        print("Use option 1 to ingest an image containing this object.")
        return

    print("\nMatching database records:")

    for row in results:
        print(
            f"ID: {row[0]} | "
            f"File: {row[1]} | "
            f"Status: {row[2]} | "
            f"Label: {row[4]} | "
            f"Confidence: {row[5]}"
        )

def browse_database():
    create_tables()

    print("\nBrowse database")
    print("1. All records")
    print("2. Filter by status")
    print("3. Image lineage")
    print("4. Attack logs")
    print("5. Filter logs")
    print("6. Back")

    choice = input("Choose an option: ").strip()

    try:
        if choice in {"1", "2"}:
            rows = list_images()

            if choice == "2":
                status = input(
                    "Status (clean / poisoned / restored / flagged / "
                    "unverified / reference): "
                ).strip().lower()

                rows = [
                    row for row in rows
                    if row[2] == status
                ]

            if not rows:
                print("No matching records.")
                return

            for row in rows:
                print(
                    f"ID: {row[0]} | "
                    f"Status: {row[2]} | "
                    f"Parent: {row[3]} | "
                    f"Label: {row[4]} | "
                    f"Confidence: {row[5]} | "
                    f"File: {row[1]}"
                )

        elif choice in {"3", "4", "5"}:
            entered_id = input("Enter image ID: ").strip()

            if not entered_id.isdigit():
                print("Enter a numeric ID.")
                return

            image_id = int(entered_id)

            if choice == "3":
                rows = get_image_lineage(image_id)

                if not rows:
                    print("Image record not found.")
                    return

                print("\nLineage: original ancestor first")

                for row in rows:
                    print(
                        f"ID: {row[0]} | "
                        f"Status: {row[2]} | "
                        f"Parent: {row[3]} | "
                        f"File: {row[1]}"
                    )

            else:
                log_type = "attack" if choice == "4" else "filter"
                rows = get_audit_logs(image_id, log_type)

                if not rows:
                    print("No logs for this image.")
                    return

                if log_type == "attack":
                    for attack, target, epsilon, reference, timestamp in rows:
                        print("\nAttack:", attack)
                        print("Target:", target)
                        print("Epsilon (0–255 units):", epsilon)
                        print("Reference ID:", reference)
                        print("Timestamp:", timestamp)
                else:
                    for step, operation, parameters in rows:
                        print(
                            f"Step {step}: {operation} | "
                            f"Parameters: {parameters}"
                        )

        elif choice != "6":
            print("Invalid option.")

    except Exception as error:
        print("Browse failed:", error)

def show_menu():
    print("\nAdversarial Image Tool")
    print("1. Ingest image")
    print("2. Diagnose image")
    print("3. Search images")
    print("4. Browse database")
    print("5. Run original baseline experiments")
    print("6. Exit")

def select_target_reference():
    target_label = input(
        "Enter target object label: "
    ).strip().lower()

    candidates = [
        row
        for row in search_images_by_label(target_label)
        if row[2] == "reference"
        and Path(row[1]).is_file()
    ]

    if not candidates:
        print("No approved reference images for:", target_label)
        return None

    print("\nApproved reference images:")

    for row in candidates:
        print(
            f"ID: {row[0]} | "
            f"File: {row[1]} | "
            f"Confidence: {row[5]:.4f}"
        )

    selected = input("Enter reference ID: ").strip()

    if not selected.isdigit():
        print("Please enter a numeric ID.")
        return None

    reference = next(
        (row for row in candidates if row[0] == int(selected)),
        None
    )

    if reference is None:
        print("Choose an ID from the displayed candidates.")
        return None

    return target_label, reference

def diagnose_selected_image():
    entered_path = input("Enter image path: ").strip().strip("\"'")
    image_path = Path(entered_path).expanduser()

    if not image_path.is_file():
        print("Image file not found:", image_path)
        return

    candidates = sorted(
        str(path)
        for folder in ["data/clean", "data/reference"]
        for path in Path(folder).rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.resolve() != image_path.resolve()
    )

    if not candidates:
        print("No reference images available for CBIR diagnosis.")
        return

    try:
        mismatch = cbir_label_mismatch(
            str(image_path),
            candidates,
            top_k=5
        )
        blur = blur_stability(str(image_path))

        flagged = bool(
            mismatch["mismatch"] and blur["changed"]
        )

        print("\nDiagnosis:")
        print("YOLO label:", mismatch["yolo_label"])
        print("CBIR majority label:", mismatch["cbir_label"])
        print("CBIR mismatch:", mismatch["mismatch"])
        print("Label after blur:", blur["blurred_label"])
        print("Prediction changed:", blur["changed"])
        print(
            "Decision:",
            "POSSIBLY POISONED"
            if flagged
            else "INSUFFICIENT EVIDENCE OF POISONING"
        )

        if flagged:
            answer = input(
                "\nTry restoring this image? y/n: "
            ).strip().lower()

            if answer == "y":
                output_path = (
                    Path("data/restored")
                    / f"{image_path.stem}_{uuid4().hex[:8]}.png"
                )

                restored, filter_log = restore_image(
                    str(image_path),
                    str(output_path)
                )

                restored_detections = detect_image(
                    str(output_path)
                )
                restored_label, restored_confidence = (
                    get_best_detection(restored_detections)
                )

                print("\nRestoration result:")
                print(
                    "Before:",
                    mismatch["yolo_label"],
                    round(mismatch["yolo_confidence"], 4)
                )
                print(
                    "After:",
                    restored_label,
                    round(restored_confidence, 4)
                )
                print("Saved image:", output_path)
                print("Filter log:", filter_log)

                parent_id, restored_id = store_restoration(
                    input_path=str(image_path),
                    output_path=str(output_path),
                    original_detections=detect_image(
                        str(image_path)
                    ),
                    restored_detections=restored_detections,
                    original_descriptor=extract_histogram(
                        str(image_path)
                    ),
                    restored_descriptor=extract_histogram(
                        str(output_path)
                    ),
                    filter_log=filter_log
                )

                print("Input database ID:", parent_id)
                print("Restored database ID:", restored_id)
                print("Parent relationship and filter log stored.")

        else:
            answer = input(
                "\nImage was not flagged. "
                "Try a targeted attack? y/n: "
            ).strip().lower()

            if answer == "y":
                selection = select_target_reference()

                if selection is not None:
                    target_label, reference = selection

                    print("\nTarget:", target_label)
                    print("Selected reference ID:", reference[0])
                    print("Reference file:", reference[1])
                    source_label = mismatch["yolo_label"]

                    if source_label == "none":
                        print("No source object detected.")
                        return

                    if source_label == target_label:
                        print("Choose a target different from the source.")
                        return

                    output_path = (
                        Path("data/poisoned")
                        / f"{image_path.stem}_reference_pgd_"
                          f"{uuid4().hex[:8]}.png"
                    )

                    targeted_pgd(
                        clean_path=str(image_path),
                        output_path=str(output_path),
                        epsilon=8,
                        steps=80,
                        step_size=1,
                        target_label=target_label,
                        reference_path=reference[1],
                        source_label=source_label
                    )

                    attacked_detections = detect_image(
                        str(output_path)
                    )

                    success, target_confidence = (
                        evaluate_attack_success(
                            attacked_detections,
                            target_label=target_label,
                            confidence_threshold=0.60
                        )
                    )

                    print("\nAttack result:")
                    print("Original label:", source_label)
                    print("Target label:", target_label)
                    print(
                        "Target confidence:",
                        round(target_confidence, 4)
                    )
                    print("Target criterion met:", success)
                    print("Saved image:", output_path)

                    parent_id, attacked_id = store_attack(
                        input_path=str(image_path),
                        output_path=str(output_path),
                        original_detections=detect_image(
                            str(image_path)
                        ),
                        attacked_detections=attacked_detections,
                        original_descriptor=extract_histogram(
                            str(image_path)
                        ),
                        attacked_descriptor=extract_histogram(
                            str(output_path)
                        ),
                        target_label=target_label,
                        reference_id=reference[0],
                        epsilon=8
                    )

                    print("Input database ID:", parent_id)
                    print("Attack database ID:", attacked_id)
                    print("Attack metadata and parent relationship stored.")

    except Exception as error:
        print("Diagnosis failed:", error)

def main():
    while True:
        show_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            ingest_image()

        elif choice == "2":
            diagnose_selected_image()

        elif choice == "3":
            search_images()

        elif choice == "4":
            browse_database()

        elif choice == "5":
            confirm = input(
                "Rerun baseline attacks and overwrite "
                "experiments/experiments.csv? y/n: "
            ).strip().lower()

            if confirm == "y":
                run_experiments()

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Enter a number from 1 to 6.")


if __name__ == "__main__":
    main()