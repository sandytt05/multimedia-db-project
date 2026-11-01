import sqlite3
import os
import json

DB_PATH = "database/images.db"


def connect_database():
    os.makedirs("database", exist_ok=True)

    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            status TEXT NOT NULL,
            parent_id INTEGER,
            detected_label TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(parent_id) REFERENCES images(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            attack_type TEXT,
            target_label TEXT,
            epsilon REAL,
            reference_image_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(image_id) REFERENCES images(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filter_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            step_number INTEGER,
            operation TEXT,
            parameters TEXT,
            FOREIGN KEY(image_id) REFERENCES images(id)
        )
    """)

    conn.commit()
    conn.close()

def insert_image(
    filepath,
    status,
    detected_label,
    confidence,
    parent_id=None
):
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO images (
            filepath,
            status,
            parent_id,
            detected_label,
            confidence
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        filepath,
        status,
        parent_id,
        detected_label,
        confidence
    ))

    image_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return image_id

def list_images():
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM images")

    rows = cursor.fetchall()

    conn.close()

    return rows

def get_image_lineage(image_id):
    conn = connect_database()
    cursor = conn.cursor()

    lineage = []

    current_id = image_id

    while current_id is not None:
        cursor.execute("""
            SELECT
                id,
                filepath,
                status,
                parent_id,
                detected_label,
                confidence
            FROM images
            WHERE id = ?
        """, (current_id,))

        row = cursor.fetchone()

        if row is None:
            break

        lineage.append(row)

        current_id = row[3]

    conn.close()

    lineage.reverse()

    return lineage

def search_images_by_label(label):
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            filepath,
            status,
            parent_id,
            detected_label,
            confidence,
            timestamp
        FROM images
        WHERE LOWER(detected_label) = LOWER(?)
    """, (label,))

    rows = cursor.fetchall()

    conn.close()

    return rows

# if __name__ == "__main__":
#     create_tables()

#     image_id = insert_image(
#         "data/clean/car1.jpg",
#         "clean",
#         "car",
#         0.9169
#     )

#     print("Image inserted with ID:", image_id)

#     for row in list_images():
#         print(row)

# if __name__ == "__main__":
#     create_tables()

#     clean_id = insert_image(
#         "data/clean/car1.jpg",
#         "clean",
#         "car",
#         0.9169
#     )

#     poisoned_id = insert_image(
#         "data/poisoned/car1_clock_eps8.jpg",
#         "poisoned",
#         "car",
#         0.9195,
#         parent_id=clean_id
#     )

#     restored_id = insert_image(
#         "data/restored/car1_clock_restored.jpg",
#         "restored",
#         "car",
#         0.9236,
#         parent_id=poisoned_id
#     )

#     print("Clean ID:", clean_id)
#     print("Poisoned ID:", poisoned_id)
#     print("Restored ID:", restored_id)

#     print()
#     print("Image lineage:")

#     lineage = get_image_lineage(restored_id)

#     for row in lineage:
#         print(row)

# if __name__ == "__main__":
#     create_tables()

#     for row in list_images():
#         print(row)

def insert_ingested_image(filepath, detections, descriptor):
    create_tables()
    conn = connect_database()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_features (
                image_id INTEGER PRIMARY KEY,
                descriptor_type TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)

        best = max(
            detections,
            key=lambda item: item["confidence"],
            default=None
        )

        cursor.execute("""
            INSERT INTO images (
                filepath, status, detected_label, confidence
            )
            VALUES (?, ?, ?, ?)
        """, (
            filepath,
            "unverified",
            best["label"] if best else "none",
            best["confidence"] if best else 0.0
        ))

        image_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO image_features (
                image_id,
                descriptor_type,
                descriptor_json,
                detections_json
            )
            VALUES (?, ?, ?, ?)
        """, (
            image_id,
            "HSV_HS_16x16",
            json.dumps(descriptor.tolist()),
            json.dumps(detections)
        ))

        conn.commit()
        return image_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

def store_restoration(
    input_path,
    output_path,
    original_detections,
    restored_detections,
    original_descriptor,
    restored_descriptor,
    filter_log
):
    create_tables()
    conn = connect_database()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_features (
                image_id INTEGER PRIMARY KEY,
                descriptor_type TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)

        # Reuse the input record if its filepath is already stored.
        cursor.execute(
            "SELECT id FROM images WHERE filepath = ? ORDER BY id LIMIT 1",
            (input_path,)
        )
        existing = cursor.fetchone()

        def add_record(path, status, detections, parent_id=None):
            best = max(
                detections,
                key=lambda item: item["confidence"],
                default=None
            )

            cursor.execute("""
                INSERT INTO images (
                    filepath, status, parent_id,
                    detected_label, confidence
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                path,
                status,
                parent_id,
                best["label"] if best else "none",
                best["confidence"] if best else 0.0
            ))
            return cursor.lastrowid

        if existing:
            parent_id = existing[0]
        else:
            parent_id = add_record(
                input_path, "flagged", original_detections
            )

        restored_id = add_record(
            output_path,
            "restored",
            restored_detections,
            parent_id
        )

        for image_id, descriptor, detections in [
            (parent_id, original_descriptor, original_detections),
            (restored_id, restored_descriptor, restored_detections),
        ]:
            cursor.execute("""
                INSERT OR REPLACE INTO image_features (
                    image_id, descriptor_type,
                    descriptor_json, detections_json
                )
                VALUES (?, ?, ?, ?)
            """, (
                image_id,
                "HSV_HS_16x16",
                json.dumps(descriptor.tolist()),
                json.dumps(detections)
            ))

        for number, step in enumerate(filter_log, start=1):
            parameters = {
                key: value
                for key, value in step.items()
                if key != "operation"
            }

            cursor.execute("""
                INSERT INTO filter_logs (
                    image_id, step_number, operation, parameters
                )
                VALUES (?, ?, ?, ?)
            """, (
                restored_id,
                number,
                step["operation"],
                json.dumps(parameters)
            ))

        conn.commit()
        return parent_id, restored_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

def mark_as_reference(image_id):
    conn = connect_database()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE images
            SET status = 'reference'
            WHERE id = ?
              AND status = 'unverified'
        """, (image_id,))

        if cursor.rowcount != 1:
            raise ValueError(
                "Record not found or its status is not unverified."
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

def store_attack(
    input_path,
    output_path,
    original_detections,
    attacked_detections,
    original_descriptor,
    attacked_descriptor,
    target_label,
    reference_id,
    epsilon=8
):
    create_tables()
    conn = connect_database()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM images
            WHERE id = ? AND status = 'reference'
        """, (reference_id,))

        if cursor.fetchone() is None:
            raise ValueError("Reference ID is not an approved reference.")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_features (
                image_id INTEGER PRIMARY KEY,
                descriptor_type TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)

        def add_record(path, status, detections, parent_id=None):
            best = max(
                detections,
                key=lambda item: item["confidence"],
                default=None
            )

            cursor.execute("""
                INSERT INTO images (
                    filepath, status, parent_id,
                    detected_label, confidence
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                path,
                status,
                parent_id,
                best["label"] if best else "none",
                best["confidence"] if best else 0.0
            ))

            return cursor.lastrowid

        cursor.execute("""
            SELECT id FROM images
            WHERE filepath = ?
            ORDER BY id LIMIT 1
        """, (input_path,))

        existing = cursor.fetchone()

        if existing:
            parent_id = existing[0]
        else:
            parent_id = add_record(
                input_path, "unverified", original_detections
            )

        attacked_id = add_record(
            output_path,
            "poisoned",
            attacked_detections,
            parent_id
        )

        for image_id, descriptor, detections in [
            (parent_id, original_descriptor, original_detections),
            (attacked_id, attacked_descriptor, attacked_detections),
        ]:
            cursor.execute("""
                INSERT OR REPLACE INTO image_features (
                    image_id, descriptor_type,
                    descriptor_json, detections_json
                )
                VALUES (?, ?, ?, ?)
            """, (
                image_id,
                "HSV_HS_16x16",
                json.dumps(descriptor.tolist()),
                json.dumps(detections)
            ))

        cursor.execute("""
            INSERT INTO attacks (
                image_id, attack_type, target_label,
                epsilon, reference_image_id
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            attacked_id,
            "reference_guided_PGD_steps80_alpha1_seed0_KL0.1",
            target_label,
            epsilon,
            reference_id
        ))

        conn.commit()
        return parent_id, attacked_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()        

def get_audit_logs(image_id, log_type):
    queries = {
        "attack": """
            SELECT attack_type, target_label, epsilon,
                   reference_image_id, timestamp
            FROM attacks
            WHERE image_id = ?
            ORDER BY id
        """,
        "filter": """
            SELECT step_number, operation, parameters
            FROM filter_logs
            WHERE image_id = ?
            ORDER BY step_number, id
        """
    }

    if log_type not in queries:
        raise ValueError("Unknown log type.")

    conn = connect_database()

    try:
        return conn.execute(
            queries[log_type], (image_id,)
        ).fetchall()
    finally:
        conn.close()

if __name__ == "__main__":
    create_tables()

    label = input("Enter label to search: ")
    results = search_images_by_label(label)

    print("\nSearch label:", label)

    if not results:
        print("No matching images found.")
    else:
        print("Matching images:")

        for row in results:
            print()
            print("ID:", row[0])
            print("File:", row[1])
            print("Status:", row[2])
            print("Parent ID:", row[3])
            print("Label:", row[4])
            print("Confidence:", row[5])
            print("Timestamp:", row[6])