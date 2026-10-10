from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def detect_image(image_path):
    results = model(image_path)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    return detections

def evaluate_attack_success(
    detections,
    target_label="clock",
    confidence_threshold=0.60
):
    target_detections = [
        detection
        for detection in detections
        if detection["label"] == target_label
    ]

    if not target_detections:
        return False, 0.0

    best_target = max(
        target_detections,
        key=lambda detection: detection["confidence"]
    )

    target_confidence = best_target["confidence"]
    success = target_confidence >= confidence_threshold

    return success, target_confidence

if __name__ == "__main__":
    epsilons = [2, 4, 6, 8]

    for epsilon in epsilons:
        image_path = (
            f"data/poisoned/"
            f"car1_clock_eps{epsilon}.jpg"
        )

        detections = detect_image(image_path)

        print()
        print("Epsilon:", epsilon)
        print("Image:", image_path)

        if not detections:
            print("No objects detected.")
        else:
            for detection in detections:
                print("Detected object:", detection["label"])
                print(
                    "Confidence:",
                    round(detection["confidence"], 4)
                )
                print("Bounding box:", detection["bbox"])

        success, target_confidence = evaluate_attack_success(
            detections,
            target_label="clock",
            confidence_threshold=0.60
        )

        print("Target class: clock")
        print("Target confidence threshold: 0.60")
        print(
            "Target confidence:",
            round(target_confidence, 4)
        )
        print("Attack successful:", success)