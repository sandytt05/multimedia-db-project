import cv2
import numpy as np
import os


def create_targeted_perturbation(
    clean_path,
    reference_path,
    output_path,
    epsilon=4
):
    clean = cv2.imread(clean_path)
    reference = cv2.imread(reference_path)

    if clean is None:
        raise ValueError("Clean image could not be loaded.")

    if reference is None:
        raise ValueError("Reference image could not be loaded.")

    reference = cv2.resize(
        reference,
        (clean.shape[1], clean.shape[0])
    )

    clean_float = clean.astype(np.float32)
    reference_float = reference.astype(np.float32)

    direction = np.sign(reference_float - clean_float)

    perturbation = epsilon * direction

    poisoned = clean_float + perturbation

    poisoned = np.clip(poisoned, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, poisoned)

    return poisoned

def create_difference_map(
    clean_path,
    poisoned_path,
    output_path,
    amplification=20
):
    clean = cv2.imread(clean_path)
    poisoned = cv2.imread(poisoned_path)

    if clean is None:
        raise ValueError("Clean image could not be loaded.")

    if poisoned is None:
        raise ValueError("Poisoned image could not be loaded.")

    clean = clean.astype(np.float32)
    poisoned = poisoned.astype(np.float32)

    difference = np.abs(poisoned - clean)
    amplified = difference * amplification

    amplified = np.clip(
        amplified,
        0,
        255
    ).astype(np.uint8)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(output_path, amplified)

if __name__ == "__main__":
    epsilons = [2, 4, 6, 8]

    for epsilon in epsilons:
        output = (
            f"data/poisoned/"
            f"car1_clock_eps{epsilon}.jpg"
        )

        create_targeted_perturbation(
            "data/clean/car1.jpg",
            "data/reference/clock1.jpg",
            output,
            epsilon=epsilon
        )

        print("Created:", output)

    create_difference_map(
        "data/clean/car1.jpg",
        "data/poisoned/car1_clock_eps4.jpg",
        "results/difference_maps/car1_clock_eps4_diff.jpg"
    )

    print(
        "Created difference map:",
        "results/difference_maps/car1_clock_eps4_diff.jpg"
    )