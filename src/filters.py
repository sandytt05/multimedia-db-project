import csv
import os

import cv2
import numpy as np

from detector import detect_image
from metrics import calculate_ssim


def gaussian_filter(
    image,
    kernel_size=3,
    sigma=0
):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma
    )


def median_filter(
    image,
    kernel_size=3
):
    return cv2.medianBlur(
        image,
        kernel_size
    )


def bilateral_filter(image):
    return cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )


def save_image(image, output_path):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    if not cv2.imwrite(output_path, image):
        raise ValueError(
            f"Could not save image: {output_path}"
        )


def sobel_edges(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobel_x ** 2
        + sobel_y ** 2
    )

    magnitude = np.clip(
        magnitude,
        0,
        255
    ).astype(np.uint8)

    return magnitude


def laplacian_edges(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    laplacian = np.absolute(
        laplacian
    )

    return np.clip(
        laplacian,
        0,
        255
    ).astype(np.uint8)


def morphological_opening(
    image,
    kernel_size=3
):
    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )


def morphological_closing(
    image,
    kernel_size=3
):
    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel
    )


def get_best_detection(detections):
    if not detections:
        return "none", 0.0

    best = max(
        detections,
        key=lambda detection: detection["confidence"]
    )

    return best["label"], best["confidence"]


if __name__ == "__main__":
    clean_path = "data/clean/car1.jpg"
    poisoned_path = (
        "data/poisoned/car1_clock_eps4.jpg"
    )

    image = cv2.imread(poisoned_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {poisoned_path}"
        )

    # Step 17: Sobel edge detection
    sobel = sobel_edges(image)

    save_image(
        sobel,
        "results/filters/car1_sobel.jpg"
    )

    print("Sobel edge image saved.")

    # Step 18: Laplacian edge detection
    laplacian = laplacian_edges(image)

    save_image(
        laplacian,
        "results/filters/car1_laplacian.jpg"
    )

    print("Laplacian edge image saved.")

    # Step 19: Morphological operations on edges
    sobel_opening = morphological_opening(
        sobel,
        3
    )

    sobel_closing = morphological_closing(
        sobel,
        3
    )

    laplacian_opening = morphological_opening(
        laplacian,
        3
    )

    laplacian_closing = morphological_closing(
        laplacian,
        3
    )

    save_image(
        sobel_opening,
        "results/filters/car1_sobel_opening3.jpg"
    )

    save_image(
        sobel_closing,
        "results/filters/car1_sobel_closing3.jpg"
    )

    save_image(
        laplacian_opening,
        (
            "results/filters/"
            "car1_laplacian_opening3.jpg"
        )
    )

    save_image(
        laplacian_closing,
        (
            "results/filters/"
            "car1_laplacian_closing3.jpg"
        )
    )

    print(
        "Morphological opening and closing "
        "images saved."
    )

    os.makedirs(
        "results/reports",
        exist_ok=True
    )

    # Steps 14–16: Filter parameter sweep
    filter_results = []

    for kernel in [3, 5, 7]:
        gaussian = gaussian_filter(
            image,
            kernel
        )

        gaussian_path = (
            f"results/filters/"
            f"car1_gaussian{kernel}.jpg"
        )

        save_image(
            gaussian,
            gaussian_path
        )

        filter_results.append(
            ("Gaussian", kernel, gaussian_path)
        )

        median = median_filter(
            image,
            kernel
        )

        median_path = (
            f"results/filters/"
            f"car1_median{kernel}.jpg"
        )

        save_image(
            median,
            median_path
        )

        filter_results.append(
            ("Median", kernel, median_path)
        )

    bilateral = bilateral_filter(image)

    bilateral_path = (
        "results/filters/car1_bilateral.jpg"
    )

    save_image(
        bilateral,
        bilateral_path
    )

    filter_results.append(
        ("Bilateral", "N/A", bilateral_path)
    )

    filter_report_path = (
        "results/reports/filter_results.csv"
    )

    with open(
        filter_report_path,
        "w",
        newline=""
    ) as report_file:
        writer = csv.writer(
    report_file,
    lineterminator="\n"
)

        writer.writerow([
            "filter",
            "kernel",
            "recovered_label",
            "confidence",
            "car_recovered",
            "ssim"
        ])

        for (
            filter_name,
            kernel,
            filtered_path
        ) in filter_results:
            detections = detect_image(
                filtered_path
            )

            label, confidence = (
                get_best_detection(detections)
            )

            ssim = calculate_ssim(
                clean_path,
                filtered_path
            )

            car_recovered = (
                label == "car"
                and confidence >= 0.60
            )

            writer.writerow([
                filter_name,
                kernel,
                label,
                round(confidence, 4),
                car_recovered,
                round(ssim, 4)
            ])

            print()
            print("Filter:", filter_name)
            print("Kernel:", kernel)
            print("Recovered label:", label)
            print(
                "Confidence:",
                round(confidence, 4)
            )
            print(
                "Car recovered:",
                car_recovered
            )
            print("SSIM:", round(ssim, 4))

    print()
    print(
        "Filter results saved to:",
        filter_report_path
    )

    # Step 20: Test restoration pipelines
    print()
    print("Testing restoration pipelines...")

    gaussian3 = gaussian_filter(
        image,
        3
    )

    median3 = median_filter(
        image,
        3
    )

    restoration_candidates = [
        (
            "Gaussian 3x3",
            gaussian3,
            "data/restored/car1_gaussian3.jpg"
        ),
        (
            "Gaussian 3x3 + Opening 3x3",
            morphological_opening(
                gaussian3,
                3
            ),
            (
                "data/restored/"
                "car1_gaussian3_opening3.jpg"
            )
        ),
        (
            "Gaussian 3x3 + Closing 3x3",
            morphological_closing(
                gaussian3,
                3
            ),
            (
                "data/restored/"
                "car1_gaussian3_closing3.jpg"
            )
        ),
        (
            "Median 3x3",
            median3,
            "data/restored/car1_median3.jpg"
        ),
        (
            "Median 3x3 + Opening 3x3",
            morphological_opening(
                median3,
                3
            ),
            (
                "data/restored/"
                "car1_median3_opening3.jpg"
            )
        ),
        (
            "Median 3x3 + Closing 3x3",
            morphological_closing(
                median3,
                3
            ),
            (
                "data/restored/"
                "car1_median3_closing3.jpg"
            )
        )
    ]

    restoration_report_path = (
        "results/reports/"
        "restoration_pipeline_results.csv"
    )

    successful_candidates = []

    with open(
        restoration_report_path,
        "w",
        newline=""
    ) as report_file:
        writer = csv.writer(
    report_file,
    lineterminator="\n"
)

        writer.writerow([
            "pipeline",
            "recovered_label",
            "confidence",
            "car_recovered",
            "ssim"
        ])

        for (
            pipeline_name,
            restored_image,
            restored_path
        ) in restoration_candidates:
            save_image(
                restored_image,
                restored_path
            )

            detections = detect_image(
                restored_path
            )

            label, confidence = (
                get_best_detection(detections)
            )

            ssim = calculate_ssim(
                clean_path,
                restored_path
            )

            car_recovered = (
                label == "car"
                and confidence >= 0.60
            )

            writer.writerow([
                pipeline_name,
                label,
                round(confidence, 4),
                car_recovered,
                round(ssim, 4)
            ])

            print()
            print("Pipeline:", pipeline_name)
            print("Recovered label:", label)
            print(
                "Confidence:",
                round(confidence, 4)
            )
            print(
                "Car recovered:",
                car_recovered
            )
            print("SSIM:", round(ssim, 4))

            if car_recovered:
                successful_candidates.append({
                    "pipeline": pipeline_name,
                    "confidence": confidence,
                    "ssim": ssim
                })

    if successful_candidates:
        best_pipeline = max(
            successful_candidates,
            key=lambda candidate: (
                candidate["ssim"],
                candidate["confidence"]
            )
        )

        print()
        print(
            "Selected best pipeline:",
            best_pipeline["pipeline"]
        )
        print(
            "Selected confidence:",
            round(
                best_pipeline["confidence"],
                4
            )
        )
        print(
            "Selected SSIM:",
            round(
                best_pipeline["ssim"],
                4
            )
        )
    else:
        print()
        print(
            "No pipeline recovered the car "
            "with confidence >= 0.60."
        )

    print()
    print(
        "Restoration results saved to:",
        restoration_report_path
    )