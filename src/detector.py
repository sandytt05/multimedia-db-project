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


if __name__ == "__main__":
    detections = detect_image("data/poisoned/car1_clock_eps4.jpg")

    for detection in detections:
        print("Detected object:", detection["label"])
        print("Confidence:", round(detection["confidence"], 4))
        print("Bounding box:", detection["bbox"])