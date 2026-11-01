from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


def targeted_pgd(
    clean_path,
    output_path,
    epsilon=8,
    steps=80,
    step_size=1,
    target_label="clock",
    reference_path=None,
    source_label="car",
):
    torch.manual_seed(0)
    np.random.seed(0)

    image = cv2.imread(clean_path)
    if image is None:
        raise ValueError(f"Could not load: {clean_path}")

    if Path(output_path).suffix.lower() != ".png":
        raise ValueError("Use PNG to preserve the pixel-change bound.")

    detector = YOLO("yolov8n.pt")
    model = detector.model.cpu().float().eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_id = next(
        key for key, value in model.names.items()
        if value == target_label
    )
    source_id = next(
        key for key, value in model.names.items()
        if value == source_label
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float() / 255.0
    )

    # Optimize original-resolution pixels, keeping the saved image bounded.
    adversarial = original.clone()
    bound = epsilon / 255.0
    alpha = step_size / 255.0

    # Match the normal detector's aspect-preserving resize
    # and padding, with a longest side of 640.
    height, width = image.shape[:2]
    stride = int(model.stride.max().item())

    scale = 640 / max(height, width)
    resized_height = round(height * scale)
    resized_width = round(width * scale)

    padding_height = (640 - resized_height) % stride
    padding_width = (640 - resized_width) % stride

    top = round(padding_height / 2 - 0.1)
    bottom = round(padding_height / 2 + 0.1)
    left = round(padding_width / 2 - 0.1)
    right = round(padding_width / 2 + 0.1)

    print(
        "Attack input size:",
        resized_height + top + bottom,
        "x",
        resized_width + left + right,
    )

    def predict_raw(tensor):
        resized = F.interpolate(
            tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

        padded = F.pad(
            resized,
            (left, right, top, bottom),
            value=114 / 255.0,
        )

        output = model(padded)
        return output[0] if isinstance(output, (tuple, list)) else output

    # Select candidates where the original model most strongly predicts car.
    with torch.no_grad():
        baseline = predict_raw(original)
        candidates = baseline[0, 4 + source_id].topk(20).indices

        reference_distribution = None

    if reference_path is not None:
        reference_image = cv2.imread(reference_path)

        if reference_image is None:
            raise ValueError(
                f"Could not load reference: {reference_path}"
            )

        # Confirm the reference contains the requested target.
        reference_results = detector.predict(
            reference_image,
            device="cpu",
            verbose=False
        )

        target_found = any(
            int(box.cls[0]) == target_id
            and float(box.conf[0]) >= 0.60
            for result in reference_results
            for box in result.boxes
        )

        if not target_found:
            raise ValueError(
                "Reference must contain the target "
                "with confidence >= 0.60."
            )

        reference_rgb = cv2.cvtColor(
            reference_image, cv2.COLOR_BGR2RGB
        )
        reference_tensor = (
            torch.from_numpy(reference_rgb.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float() / 255.0
        )

        with torch.no_grad():
            reference_predictions = predict_raw(reference_tensor)
            reference_index = reference_predictions[
                0, 4 + target_id
            ].argmax()

            reference_scores = reference_predictions[
                0, 4:, reference_index
            ].clamp_min(1e-8)

            reference_distribution = (
                reference_scores / reference_scores.sum()
            ).detach()

        print("Reference guide:", reference_path)

    for step in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        predictions = predict_raw(adversarial)

        target_scores = predictions[0, 4 + target_id, candidates]
        source_scores = predictions[0, 4 + source_id, candidates]

        # Increase clock scores and decrease car scores.
        objective = (
            torch.log(target_scores.clamp_min(1e-8))
            - torch.log(source_scores.clamp_min(1e-8))
        ).mean()

        if reference_distribution is not None:
            class_scores = predictions[
                0, 4:, candidates
            ].clamp_min(1e-8)

            distributions = (
                class_scores
                / class_scores.sum(dim=0, keepdim=True)
            )

            reference_loss = F.kl_div(
                distributions.log().transpose(0, 1),
                reference_distribution.unsqueeze(0).expand(
                    candidates.numel(), -1
                ),
                reduction="batchmean"
            )

            objective = objective - 0.1 * reference_loss

        gradient = torch.autograd.grad(objective, adversarial)[0]

        with torch.no_grad():
            adversarial = adversarial + alpha * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, original + bound),
                original - bound,
            ).clamp(0, 1)

        if (step + 1) % 10 == 0:
            print(
                f"Step {step + 1}/{steps} | "
                f"Target score: {target_scores.max().item():.4f}"
            )

    output_rgb = (
        adversarial.detach()[0]
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .clamp(0, 255)
        .byte()
        .numpy()
    )
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, output_bgr):
        raise IOError(f"Could not save: {output_path}")

    saved = cv2.imread(output_path)
    actual_epsilon = int(
        np.abs(saved.astype(np.int16) - image.astype(np.int16)).max()
    )
    print("\nSaved:", output_path)
    print("Actual maximum pixel change:", actual_epsilon)
    assert actual_epsilon <= epsilon


if __name__ == "__main__":
    targeted_pgd(
        clean_path="data/clean/car1.jpg",
        output_path="data/poisoned/car1_clock_pgd_eps8.png",
    )