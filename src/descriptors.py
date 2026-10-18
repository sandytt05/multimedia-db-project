import cv2
import numpy as np
import glob

def extract_histogram(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR image from OpenCV into HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Build a 2D histogram using:
    # Hue = 16 bins
    # Saturation = 16 bins
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256]
    )

    # Normalize so different image sizes can be compared fairly
    cv2.normalize(
        histogram,
        histogram
    )

    # Turn 16x16 histogram into one flat vector
    descriptor = histogram.flatten()

    return descriptor

def compare_histograms(histogram1, histogram2):
    histogram1 = histogram1.astype(np.float32)
    histogram2 = histogram2.astype(np.float32)

    distance = cv2.compareHist(
        histogram1,
        histogram2,
        cv2.HISTCMP_BHATTACHARYYA
    )

    return distance

def query_by_image(query_path, image_paths, top_k=5):
    query_descriptor = extract_histogram(query_path)

    results = []

    for image_path in image_paths:
        descriptor = extract_histogram(image_path)

        distance = compare_histograms(
            query_descriptor,
            descriptor
        )

        results.append({
            "image_path": image_path,
            "distance": distance
        })

    results.sort(
        key=lambda item: item["distance"]
    )

    return results[:top_k]


# if __name__ == "__main__":
#     image_path = "data/clean/car2.jpg"

#     descriptor = extract_histogram(image_path)

#     print("Descriptor created.")
#     print("Descriptor length:", len(descriptor))
#     print("First 10 values:")
#     print(descriptor[:10])

# if __name__ == "__main__":
#     image1 = "data/clean/car1.jpg"
#     image2 = "data/clean/car2.jpg"

#     descriptor1 = extract_histogram(image1)
#     descriptor2 = extract_histogram(image2)

#     distance = compare_histograms(
#         descriptor1,
#         descriptor2
#     )

#     print("Image 1:", image1)
#     print("Image 2:", image2)
#     print(
#         "Bhattacharyya distance:",
#         round(distance, 4)
#     )

if __name__ == "__main__":
    query_path = "data/clean/car1.jpg"

    image_paths = glob.glob(
        "data/clean/*.jpg"
    )

    image_paths += glob.glob(
        "data/reference/*.jpg"
    )

    image_paths = [
        path
        for path in image_paths
        if path != query_path
    ]

    results = query_by_image(
        query_path,
        image_paths,
        top_k=5
    )

    print("Query image:", query_path)
    print()
    print("Top 5 most similar images:")

    for index, result in enumerate(
        results,
        start=1
    ):
        print(
            index,
            result["image_path"],
            "distance =",
            round(result["distance"], 4)
        )