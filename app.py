
from pathlib import Path
import sys
import json
import sqlite3
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PROJECT SETUP
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from detector import detect_image
from attack import create_targeted_perturbation
from filters import restore_image
from metrics import calculate_psnr, calculate_ssim
from descriptors import extract_histogram, compare_histograms
from database import create_tables, insert_image

UPLOAD_DIR = ROOT / "data" / "uploads"
CLEAN_DIR = ROOT / "data" / "clean"
REFERENCE_DIR = ROOT / "data" / "reference"
POISONED_DIR = ROOT / "data" / "poisoned"
RESTORED_DIR = ROOT / "data" / "restored"
DIFF_DIR = ROOT / "results" / "difference_maps"
DB_PATH = ROOT / "database" / "images.db"

for folder in [
    UPLOAD_DIR,
    CLEAN_DIR,
    REFERENCE_DIR,
    POISONED_DIR,
    RESTORED_DIR,
    DIFF_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

create_tables()

st.set_page_config(
    page_title="Adversarial Image Lab",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def save_upload(uploaded_file, directory):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in [".jpg", ".jpeg", ".png"]:
        raise ValueError("Unsupported file type.")

    filename = f"{uuid.uuid4().hex}{suffix}"
    path = directory / filename
    path.write_bytes(uploaded_file.getvalue())

    if cv2.imread(str(path)) is None:
        path.unlink(missing_ok=True)
        raise ValueError("The uploaded file is not a valid image.")

    return path


def display_image(path, caption=None):
    st.image(str(path), caption=caption, width="stretch")


def main_detection(path):
    detections = detect_image(str(path))

    if not detections:
        return None

    return max(
        detections,
        key=lambda item: item["confidence"]
    )


def show_detection(detection):
    if detection is None:
        st.warning("YOLO found no objects.")
        return

    st.write("**Detected label:**", detection["label"])
    st.write(
        "**Confidence:**",
        f'{detection["confidence"]:.4f}'
    )

    if "bbox" in detection:
        st.write("**Bounding box:**", detection["bbox"])


def image_metrics(clean_path, other_path):
    psnr = calculate_psnr(
        str(clean_path),
        str(other_path)
    )
    ssim = calculate_ssim(
        str(clean_path),
        str(other_path)
    )
    return psnr, ssim


def show_metrics(clean_path, other_path):
    try:
        psnr, ssim = image_metrics(
            clean_path,
            other_path
        )

        col1, col2 = st.columns(2)
        col1.metric("PSNR (dB)", f"{psnr:.2f}")
        col2.metric("SSIM", f"{ssim:.4f}")

    except Exception as error:
        st.warning(f"Could not calculate metrics: {error}")


def read_images():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM images ORDER BY id DESC",
            conn
        )


def save_attack_log(image_id, target, epsilon):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO attacks
            (image_id, attack_type, target_label, epsilon)
            VALUES (?, ?, ?, ?)
            """,
            (
                image_id,
                "reference-guided bounded perturbation",
                target,
                epsilon
            )
        )


def save_filter_log(image_id, steps):
    with sqlite3.connect(DB_PATH) as conn:
        for number, step in enumerate(steps, 1):
            operation = step.get("operation", "unknown")
            parameters = {
                key: value
                for key, value in step.items()
                if key != "operation"
            }

            conn.execute(
                """
                INSERT INTO filter_logs
                (image_id, step_number, operation, parameters)
                VALUES (?, ?, ?, ?)
                """,
                (
                    image_id,
                    number,
                    operation,
                    json.dumps(parameters)
                )
            )


def create_difference(clean_path, poisoned_path, output_path):
    clean = cv2.imread(str(clean_path)).astype(np.float32)
    poisoned = cv2.imread(str(poisoned_path)).astype(np.float32)

    difference = np.abs(poisoned - clean) * 20
    difference = np.clip(difference, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), difference)


def list_images(folder):
    return sorted(
        [
            path for path in folder.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ],
        key=lambda path: path.name
    )


def similar_images(query_path, top_k=5):
    query_hist = extract_histogram(str(query_path))
    records = read_images()
    results = []

    for _, record in records.iterrows():
        path = Path(record["filepath"])

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            continue

        if path.resolve() == Path(query_path).resolve():
            continue

        try:
            candidate_hist = extract_histogram(str(path))
            distance = compare_histograms(
                query_hist,
                candidate_hist
            )

            results.append({
                "id": int(record["id"]),
                "path": path,
                "label": record["detected_label"],
                "status": record["status"],
                "distance": float(distance),
            })

        except Exception:
            continue

    return sorted(
        results,
        key=lambda item: item["distance"]
    )[:top_k]


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("Adversarial Image Lab")
st.caption(
    "YOLO Detection | Image Attack | Restoration | "
    "CBIR | Multimedia Database"
)

with st.sidebar:
    st.header("Project")
    st.write("Multimedia Database System")
    st.write("Model: YOLOv8")
    st.divider()
    st.info(
        "This application uses the Python modules "
        "from the existing project."
    )

tabs = st.tabs([
    "Upload & Detect",
    "Poison Image",
    "Restore Image",
    "CBIR Search",
    "Database",
    "Diagnosis",
])

# --------------------------------------------------
# 4. UPLOAD AND DETECTION
# --------------------------------------------------

with tabs[0]:
    st.header("Upload and Detect Objects")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="clean_upload"
    )

    if uploaded is not None:
        st.image(uploaded, width="stretch")

        if st.button("Run YOLO & Save", type="primary"):
            try:
                path = save_upload(uploaded, CLEAN_DIR)
                detection = main_detection(path)

                label = detection["label"] if detection else None
                confidence = (
                    detection["confidence"] if detection else None
                )

                image_id = insert_image(
                    str(path),
                    "clean",
                    label,
                    confidence
                )

                st.session_state["clean_path"] = str(path)
                st.session_state["clean_id"] = image_id
                st.session_state["clean_detection"] = detection

            except Exception as error:
                st.error(f"Upload failed: {error}")

    if st.session_state.get("clean_path"):
        st.subheader("Saved Image")
        display_image(st.session_state["clean_path"])
        show_detection(
            st.session_state.get("clean_detection")
        )

# --------------------------------------------------
# 5. IMAGE POISONING
# --------------------------------------------------

with tabs[1]:
    st.header("Generate Adversarial Image")

    clean_path = st.session_state.get("clean_path")
    clean_id = st.session_state.get("clean_id")

    if not clean_path:
        st.warning("Upload a clean image first.")

    else:
        references = list_images(REFERENCE_DIR)

        display_image(clean_path, "Original image")

        if not references:
            st.warning(
                "Add reference images to data/reference first."
            )

        else:
            reference = st.selectbox(
                "Select reference image",
                references,
                format_func=lambda path: path.name
            )

            target = st.text_input(
                "Target YOLO class",
                value="Clock"
            ).strip()

            epsilon = st.slider(
                "Perturbation strength (pixel units)",
                min_value=1,
                max_value=8,
                value=4
            )

            display_image(reference, "Reference image")

            st.caption(
                "This reference-guided perturbation is a baseline, "
                "not a guaranteed targeted adversarial attack."
            )

            if st.button("Generate Poisoned Image"):
                try:
                    filename = f"{uuid.uuid4().hex}.png"
                    output = POISONED_DIR / filename

                    create_targeted_perturbation(
                        str(clean_path),
                        str(reference),
                        str(output),
                        epsilon=epsilon
                    )

                    detection = main_detection(output)
                    label = detection["label"] if detection else None
                    confidence = (
                        detection["confidence"] if detection else None
                    )

                    image_id = insert_image(
                        str(output),
                        "poisoned",
                        label,
                        confidence,
                        parent_id=clean_id
                    )

                    save_attack_log(image_id, target, epsilon)

                    diff_path = DIFF_DIR / filename
                    create_difference(
                        clean_path,
                        output,
                        diff_path
                    )

                    st.session_state["poisoned_path"] = str(output)
                    st.session_state["poisoned_id"] = image_id
                    st.session_state["poisoned_detection"] = detection
                    st.session_state["diff_path"] = str(diff_path)
                    st.session_state["target"] = target

                except Exception as error:
                    st.error(f"Attack failed: {error}")

    if st.session_state.get("poisoned_path"):
        st.subheader("Attack Result")

        col1, col2 = st.columns(2)

        with col1:
            display_image(
                st.session_state["poisoned_path"],
                "Poisoned image"
            )
            show_detection(
                st.session_state["poisoned_detection"]
            )

        with col2:
            display_image(
                st.session_state["diff_path"],
                "Amplified difference map"
            )

        detection = st.session_state["poisoned_detection"]
        target = st.session_state["target"]

        if (
            detection is not None
            and detection["label"].lower() == target.lower()
            and detection["confidence"] >= 0.60
        ):
            st.success("Target detection reached the success threshold.")
        else:
            st.warning("Target detection did not reach the success threshold.")

        if st.session_state.get("clean_path"):
            show_metrics(
                st.session_state["clean_path"],
                st.session_state["poisoned_path"]
            )

# --------------------------------------------------
# 6. RESTORATION
# --------------------------------------------------

with tabs[2]:
    st.header("Restore Poisoned Image")

    poisoned_path = st.session_state.get("poisoned_path")
    poisoned_id = st.session_state.get("poisoned_id")

    if not poisoned_path:
        st.warning("Generate a poisoned image first.")

    else:
        display_image(poisoned_path, "Before restoration")

        if st.button("Restore Image", type="primary"):
            try:
                output = RESTORED_DIR / f"{uuid.uuid4().hex}.png"

                restored, filter_log = restore_image(
                    str(poisoned_path),
                    str(output)
                )

                detection = main_detection(output)

                label = detection["label"] if detection else None
                confidence = (
                    detection["confidence"] if detection else None
                )

                image_id = insert_image(
                    str(output),
                    "restored",
                    label,
                    confidence,
                    parent_id=poisoned_id
                )

                save_filter_log(image_id, filter_log)

                st.session_state["restored_path"] = str(output)
                st.session_state["restored_detection"] = detection

                st.success("Restoration completed and saved.")

            except Exception as error:
                st.error(f"Restoration failed: {error}")

    if st.session_state.get("restored_path"):
        st.subheader("Restoration Result")
        display_image(
            st.session_state["restored_path"],
            "Restored image"
        )
        show_detection(
            st.session_state["restored_detection"]
        )

        if st.session_state.get("clean_path"):
            show_metrics(
                st.session_state["clean_path"],
                st.session_state["restored_path"]
            )

# --------------------------------------------------
# 7. CBIR SEARCH
# --------------------------------------------------

with tabs[3]:
    st.header("Content-Based Image Retrieval")

    search_method = st.radio(
        "Search method",
        ["Query by Image", "Query by Label"],
        horizontal=True
    )

    if search_method == "Query by Image":
        query_upload = st.file_uploader(
            "Upload a query image",
            type=["jpg", "jpeg", "png"],
            key="query_upload"
        )

        if query_upload is not None:
            st.image(query_upload, width=300)

            if st.button("Find Similar Images"):
                try:
                    query_path = save_upload(
                        query_upload,
                        UPLOAD_DIR
                    )

                    results = similar_images(query_path, 5)

                    if not results:
                        st.info("No indexed images found.")

                    for item in results:
                        col1, col2 = st.columns([1, 3])

                        with col1:
                            display_image(item["path"])

                        with col2:
                            st.write("**Label:**", item["label"])
                            st.write("**Status:**", item["status"])
                            st.write(
                                "**Distance:**",
                                f'{item["distance"]:.4f}'
                            )

                except Exception as error:
                    st.error(f"Search failed: {error}")

    else:
        label_query = st.text_input("Enter detected label")

        if st.button("Search by Label"):
            records = read_images()

            if label_query.strip():
                matches = records[
                    records["detected_label"]
                    .fillna("")
                    .str.lower() == label_query.strip().lower()
                ]

                if matches.empty:
                    st.info("No matching images found.")
                else:
                    st.dataframe(
                        matches,
                        width="stretch",
                        hide_index=True
                    )

# --------------------------------------------------
# 8. DATABASE BROWSER
# --------------------------------------------------

with tabs[4]:
    st.header("Multimedia Database")

    status_filter = st.selectbox(
        "Filter by status",
        ["All", "clean", "poisoned", "restored", "flagged"]
    )

    try:
        records = read_images()

        if status_filter != "All":
            records = records[
                records["status"] == status_filter
            ]

        st.metric("Images found", len(records))

        st.dataframe(
            records,
            width="stretch",
            hide_index=True
        )

        if not records.empty:
            selected_id = st.selectbox(
                "Inspect image ID",
                records["id"].tolist()
            )

            selected = records[
                records["id"] == selected_id
            ].iloc[0]

            path = Path(selected["filepath"])

            if not path.is_absolute():
                path = ROOT / path

            if path.exists():
                display_image(path)

            st.write("**Status:**", selected["status"])
            st.write("**Label:**", selected["detected_label"])
            st.write("**Parent ID:**", selected["parent_id"])

    except Exception as error:
        st.error(f"Database error: {error}")

# --------------------------------------------------
# 9. DIAGNOSTIC ANALYSIS
# --------------------------------------------------

with tabs[5]:
    st.header("Image Diagnosis")

    st.write(
        "Inspect prediction stability, visual similarity, "
        "and high-frequency image energy."
    )

    candidates = (
        list_images(CLEAN_DIR)
        + list_images(POISONED_DIR)
        + list_images(RESTORED_DIR)
    )

    if not candidates:
        st.info("No images available for diagnosis.")

    else:
        selected_path = st.selectbox(
            "Select image",
            candidates,
            format_func=lambda path: (
                f"{path.parent.name}/{path.name}"
            )
        )

        if st.button("Analyze Image"):
            try:
                image = cv2.imread(str(selected_path))
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                laplacian = cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                )

                energy = float(np.mean(np.abs(laplacian)))

                blurred = cv2.GaussianBlur(
                    image,
                    (3, 3),
                    0
                )

                blur_path = UPLOAD_DIR / (
                    f"blur_{uuid.uuid4().hex}.png"
                )
                cv2.imwrite(str(blur_path), blurred)

                original_detection = main_detection(selected_path)
                blurred_detection = main_detection(blur_path)

                blur_path.unlink(missing_ok=True)

                original_label = (
                    original_detection["label"]
                    if original_detection else None
                )
                blurred_label = (
                    blurred_detection["label"]
                    if blurred_detection else None
                )

                st.subheader("YOLO Prediction")
                show_detection(original_detection)

                st.subheader("Stability Test")
                st.write("Original prediction:", original_label)
                st.write("After Gaussian blur:", blurred_label)
                st.write(
                    "Prediction changed:",
                    original_label != blurred_label
                )

                st.metric(
                    "High-frequency energy",
                    f"{energy:.3f}"
                )

                st.subheader("Nearest Visual Neighbors")
                neighbors = similar_images(selected_path, 5)

                for neighbor in neighbors:
                    st.write(
                        neighbor["label"],
                        "— distance:",
                        f'{neighbor["distance"]:.4f}'
                    )

                st.info(
                    "These are diagnostic signals, not yet a "
                    "calibrated clean/poisoned classification. "
                    "Evaluate thresholds on labeled clean and "
                    "poisoned images before enabling an "
                    "automatic decision."
                )

            except Exception as error:
                st.error(f"Diagnosis failed: {error}")

# --------------------------------------------------
# END
# --------------------------------------------------
