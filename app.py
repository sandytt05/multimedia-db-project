from pathlib import Path
from PIL import Image, ImageOps
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
from attack_pgd import targeted_pgd
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
    preview_width = st.session_state.get("preview_width", 440)

    source = str(path) if isinstance(path, (str, Path)) else path

    if hasattr(source, "seek"):
        source.seek(0)

    with Image.open(source) as opened:
        preview = ImageOps.exif_transpose(opened).convert("RGB")

        # Only resize the displayed preview.
        # The original image file remains unchanged.
        preview.thumbnail(
            (preview_width, 280),
            Image.Resampling.LANCZOS,
        )

    st.image(
        preview,
        caption=caption,
        width=preview.width,
    )

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
                "targeted PGD",
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

st.markdown(
    """
<style>
/* Main workspace */
.stApp {
    background-color: #f8fafc;
    color: #1e293b;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #172334;
    min-width: 230px;
    max-width: 260px;
}

[data-testid="stSidebarContent"] {
    padding-top: 0;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 1rem;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #364152;
}

[data-testid="stSidebar"] button {
    color: #e2e8f0;
}

/* Compact headings */
[data-testid="stMain"] h1 {
    font-size: 2rem !important;
    font-weight: 700;
    letter-spacing: -0.04rem;
    color: #172334;
}

[data-testid="stMain"] h2 {
    font-size: 1.35rem !important;
    color: #172334;
}

[data-testid="stMain"] h3 {
    font-size: 1.1rem !important;
    color: #172334;
}

/* Navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 0.8rem;
    border-radius: 6px;
    color: #526174;
}

.stTabs [aria-selected="true"] {
    background-color: #e7edf5;
    color: #243f63 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #3d5f89 !important;
}

.stTabs [data-baseweb="tab-border"] {
    background-color: #dce3ec;
}

/* Buttons */
[data-testid="stBaseButton-primary"] {
    background-color: #304f75;
    border-color: #304f75;
    color: white;
    border-radius: 7px;
}

[data-testid="stBaseButton-primary"]:hover {
    background-color: #243f63;
    border-color: #243f63;
    color: white;
}

/* Result cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 0.7rem 0.9rem;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}

/* Compact previews */
[data-testid="stImage"] img {
    border-radius: 8px;
    max-width: 100%;
    height: auto;
}

/* Neutral upload area */
[data-testid="stFileUploaderDropzone"] {
    background-color: white;
    border: 1px dashed #b8c4d3;
    border-radius: 8px;
}
</style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Vision Workspace")
    st.caption("Multimedia Database Project")

    st.divider()

    st.markdown("**Workflow**")
    st.caption(
        "01  Upload & detect\n\n"
        "02  Generate an attack\n\n"
        "03  Restore & compare"
    )

    st.divider()

    st.markdown("**Display settings**")

    st.slider(
        "Preview width",
        min_value=280,
        max_value=600,
        value=440,
        step=40,
        key="preview_width",
        help="Changes displayed previews only.",
    )

    st.caption("Preview size does not affect image quality.")

    st.divider()

    st.caption("Sandy Taha · Perla Hoyek")
    st.caption("YOLOv8 · SQLite · CBIR")

st.title("Adversarial Image Lab")
st.caption("Sandy Taha · Perla Hoyek")
st.write("Detect objects, test targeted attacks, and compare recovery.")

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
    st.header("Upload & Detect")

    controls, preview_column = st.columns(
        [1, 1.4],
        gap="large",
    )

    with controls:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="clean_upload",
        )

        if uploaded is not None:
            if st.button(
                "Run YOLO & Save",
                type="primary",
            ):
                try:
                    path = save_upload(uploaded, CLEAN_DIR)

                    with st.spinner("Detecting objects..."):
                        detection = main_detection(path)

                    label = (
                        detection["label"]
                        if detection else None
                    )

                    confidence = (
                        detection["confidence"]
                        if detection else None
                    )

                    image_id = insert_image(
                        str(path),
                        "clean",
                        label,
                        confidence,
                    )

                    st.session_state["clean_path"] = str(path)
                    st.session_state["clean_id"] = image_id
                    st.session_state["clean_detection"] = detection

                    st.success(
                        f"Saved to database · ID {image_id}"
                    )

                except Exception as error:
                    st.error(f"Upload failed: {error}")

        if st.session_state.get("clean_path"):
            st.caption("Detection for the last saved image")
            show_detection(
                st.session_state.get("clean_detection")
            )

    with preview_column:
        if uploaded is not None:
            display_image(
                uploaded,
                "Selected image",
            )

        elif st.session_state.get("clean_path"):
            display_image(
                st.session_state["clean_path"],
                "Last saved image",
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
        display_image(clean_path, "Original image")
        references = list_images(REFERENCE_DIR)

        if not references:
            st.warning("Add reference images to data/reference first.")

        else:
            # Cache reference predictions until their files change.
            reference_cache = st.session_state.setdefault(
                "reference_detection_cache", {}
            )
            references_by_label = {}

            for reference_path in references:
                reference_path = Path(reference_path)
                cache_key = (
                    str(reference_path.resolve()),
                    reference_path.stat().st_mtime_ns,
                )

                if cache_key not in reference_cache:
                    try:
                        reference_cache[cache_key] = main_detection(
                            reference_path
                        )
                    except Exception:
                        reference_cache[cache_key] = None

                reference_detection = reference_cache[cache_key]

                if (
                    reference_detection is not None
                    and reference_detection["confidence"] >= 0.60
                ):
                    reference_label = (
                        reference_detection["label"].strip().lower()
                    )
                    references_by_label.setdefault(
                        reference_label, []
                    ).append(reference_path)

            if not references_by_label:
                st.warning(
                    "No reference image has a YOLO detection with "
                    "confidence ≥ 0.60. Add a clearer reference image."
                )

            else:
                targets = sorted(references_by_label)
                default_index = (
                    targets.index("clock") if "clock" in targets else 0
                )

                target = st.selectbox(
                    "Target YOLO class",
                    targets,
                    index=default_index,
                    key="attack_target_class",
                )

                matching_references = references_by_label[target]

                if len(matching_references) == 1:
                    reference = matching_references[0]
                    st.write(
                        "**Automatically selected reference:**",
                        reference.name,
                    )
                else:
                    reference = st.selectbox(
                        "Select matching reference image",
                        matching_references,
                        format_func=lambda path: path.name,
                        key=f"attack_reference_{target}",
                    )

                display_image(
                    reference,
                    f"Reference for target: {target}",
                )

                epsilon = st.slider(
                    "Perturbation strength (pixel units)",
                    min_value=1,
                    max_value=8,
                    value=4,
                    key="attack_epsilon",
                )

                st.caption(
                    "PGD changes the original image's pixels to "
                    "encourage the selected target label, using "
                    "the matching reference as guidance."
                )

                if st.button("Generate Poisoned Image"):
                    try:
                        source_detection = main_detection(clean_path)

                        if source_detection is None:
                            st.warning(
                                "No source object detected. "
                                "Choose another original image."
                            )

                        elif (
                            source_detection["label"].strip().lower()
                            == target
                        ):
                            st.warning(
                                "Choose a target different from "
                                "the original detected label."
                            )

                        else:
                            source_label = source_detection["label"]
                            filename = f"{uuid.uuid4().hex}.png"
                            output = POISONED_DIR / filename

                            with st.spinner(
                                f"Generating attack: "
                                f"{source_label} → {target}..."
                            ):
                                targeted_pgd(
                                    clean_path=str(clean_path),
                                    output_path=str(output),
                                    epsilon=epsilon,
                                    steps=80,
                                    step_size=1,
                                    target_label=target,
                                    reference_path=str(reference),
                                    source_label=source_label,
                                )

                                detections = detect_image(str(output))
                                detection = (
                                    max(
                                        detections,
                                        key=lambda item: item["confidence"],
                                    )
                                    if detections else None
                                )

                                target_confidence = max(
                                    (
                                        item["confidence"]
                                        for item in detections
                                        if item["label"].strip().lower()
                                        == target
                                    ),
                                    default=0.0,
                                )

                                label = (
                                    detection["label"]
                                    if detection else None
                                )
                                confidence = (
                                    detection["confidence"]
                                    if detection else None
                                )

                                image_id = insert_image(
                                    str(output),
                                    "poisoned",
                                    label,
                                    confidence,
                                    parent_id=clean_id,
                                )

                                save_attack_log(
                                    image_id, target, epsilon
                                )

                                diff_path = DIFF_DIR / filename
                                create_difference(
                                    clean_path,
                                    output,
                                    diff_path,
                                )

                            st.session_state.update({
                                "poisoned_path": str(output),
                                "poisoned_id": image_id,
                                "poisoned_detection": detection,
                                "diff_path": str(diff_path),
                                "target": target,
                                "attack_target_confidence": float(
                                    target_confidence
                                ),
                                "attack_source_path": str(clean_path),
                                "attack_source_label": source_label,
                                "attack_reference_path": str(reference),
                            })

                    except Exception as error:
                        st.error(f"Attack failed: {error}")
                        st.exception(error)

    if st.session_state.get("poisoned_path"):
        st.subheader("Last Generated Attack Result")

        st.write(
            "**Attack target:**",
            st.session_state.get("target", "Unknown"),
        )
        st.write(
            "**Original label:**",
            st.session_state.get("attack_source_label", "Unknown"),
        )

        used_reference = st.session_state.get("attack_reference_path")
        if used_reference:
            st.write("**Reference used:**", Path(used_reference).name)

        col1, col2 = st.columns(2)

        with col1:
            display_image(
                st.session_state["poisoned_path"],
                "Adversarial image",
            )
            show_detection(
                st.session_state.get("poisoned_detection")
            )

        with col2:
            display_image(
                st.session_state["diff_path"],
                "Amplified difference map",
            )

        target_confidence = st.session_state.get(
            "attack_target_confidence"
        )

        if target_confidence is not None:
            st.write(
                "**Target confidence:**",
                f"{target_confidence:.4f}",
            )

            if target_confidence >= 0.60:
                st.success(
                    "Target detection reached the success threshold."
                )
            else:
                st.warning(
                    "Target detection did not reach the success threshold."
                )
        else:
            st.info(
                "Generate a new attack to evaluate its target confidence."
            )

        source_path = st.session_state.get("attack_source_path")
        if source_path:
            show_metrics(
                source_path,
                st.session_state["poisoned_path"],
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
            display_image(query_upload, "Query image")

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
    st.write("Inspect the recorded status, YOLO prediction, and visual similarity.")

    candidates = (
        list_images(CLEAN_DIR)
        + list_images(POISONED_DIR)
        + list_images(RESTORED_DIR)
    )

    records = read_images()

    def resolve_db_path(filepath):
        if pd.isna(filepath):
            return None
        return (ROOT / Path(filepath)).resolve()

    record_paths = records["filepath"].map(resolve_db_path)
    record_ids_by_path = {}

    for image_id, path in zip(records["id"], record_paths):
        if path is not None and path.is_file():
            record_ids_by_path.setdefault(path, []).append(int(image_id))

    candidates = list(dict.fromkeys(
        [path.resolve() for path in candidates if path.is_file()]
        + list(record_ids_by_path)
    ))

    def diagnosis_option(path):
        name = f"{path.parent.name}/{path.name}"
        ids = record_ids_by_path.get(path, [])
        if ids:
            return f"ID {', '.join(map(str, ids))} — {name}"
        return name

    if not candidates:
        st.info("No images available for diagnosis.")
    else:
        selected_path = st.selectbox(
            "Select image",
            candidates,
            format_func=diagnosis_option,
            key="diagnosis_image",
        )

        if st.button("Analyze Image"):
            try:
                image = cv2.imread(str(selected_path))
                if image is None:
                    raise ValueError(f"Could not read image: {selected_path}")

                detection = main_detection(selected_path)
                current_label = detection["label"] if detection else None

                matches = records[record_paths == selected_path]
                poisoned_matches = matches[matches["status"] == "poisoned"]
                if not poisoned_matches.empty:
                    matches = poisoned_matches

                status = "Unknown (image not registered)"
                source_label = "Unknown"

                if not matches.empty:
                    record = matches.iloc[0]
                    stored_status = str(record["status"]).strip().lower()
                    status = (
                        "Possibly poisoned"
                        if stored_status == "flagged"
                        else stored_status.capitalize()
                    )

                    if stored_status == "poisoned":
                        with sqlite3.connect(DB_PATH) as conn:
                            has_outcomes = conn.execute(
                                """
                                SELECT 1 FROM sqlite_master
                                WHERE type = 'table'
                                  AND name = 'baseline_outcomes'
                                """
                            ).fetchone()

                            if has_outcomes:
                                outcome = conn.execute(
                                    """
                                    SELECT target_reached
                                    FROM baseline_outcomes
                                    WHERE image_id = ?
                                    """,
                                    (int(record["id"]),),
                                ).fetchone()

                                if outcome is not None and outcome[0] == 0:
                                    status = "Perturbed — target not reached"

                    label = None

                    if stored_status == "clean":
                        label = record["detected_label"]
                    else:
                        parent_id = record["parent_id"]
                        seen = {int(record["id"])}

                        while pd.notna(parent_id) and int(parent_id) not in seen:
                            seen.add(int(parent_id))
                            parents = records[records["id"] == int(parent_id)]
                            if parents.empty:
                                break

                            parent = parents.iloc[0]
                            if str(parent["status"]).strip().lower() == "clean":
                                label = parent["detected_label"]
                                break

                            parent_id = parent["parent_id"]

                    if pd.notna(label) and str(label).strip().lower() not in {"", "none"}:
                        source_label = str(label).strip().title()

                st.subheader("Diagnosis Result")
                st.write("**1. Status:**", status)
                st.write(
                    "**2. YOLO detects:**",
                    current_label.title() if current_label else "No object detected",
                )
                st.write("**3. Original label:**", source_label)

                if detection:
                    st.write("**Confidence:**", f'{detection["confidence"]:.4f}')
                    if "bbox" in detection:
                        st.write("**Bounding box:**", detection["bbox"])

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                energy = float(np.mean(np.abs(laplacian)))

                blurred = cv2.GaussianBlur(image, (3, 3), 0)
                blur_path = UPLOAD_DIR / f"blur_{uuid.uuid4().hex}.png"

                try:
                    if not cv2.imwrite(str(blur_path), blurred):
                        raise OSError("Could not save the blurred image.")
                    blurred_detection = main_detection(blur_path)
                finally:
                    blur_path.unlink(missing_ok=True)

                blurred_label = (
                    blurred_detection["label"] if blurred_detection else None
                )

                st.subheader("Stability Test")
                st.write("Before blur:", current_label or "No object detected")
                st.write("After Gaussian blur:", blurred_label or "No object detected")
                st.write("Prediction changed:", current_label != blurred_label)
                st.metric("High-frequency energy", f"{energy:.3f}")

                st.subheader("Nearest Visual Neighbors")
                neighbors = similar_images(selected_path, 5)

                if not neighbors:
                    st.info("No indexed neighbors found.")

                for neighbor in neighbors:
                    st.write(
                        neighbor["label"],
                        "— distance:",
                        f'{neighbor["distance"]:.4f}',
                    )

                st.caption(
                    "Status and original label come from the database. "
                    "YOLO and the stability test analyze the selected file."
                )

            except Exception as error:
                st.error(f"Diagnosis failed: {error}")