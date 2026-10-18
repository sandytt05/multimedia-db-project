import sqlite3
import os

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

if __name__ == "__main__":
    create_tables()

    label = input("Enter label to search: ")

    results = search_images_by_label(label)

    print()
    print("Search label:", label)

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