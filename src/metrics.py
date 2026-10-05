from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2


def calculate_psnr(clean_path, other_path):
    clean = cv2.imread(clean_path)
    other = cv2.imread(other_path)

    return peak_signal_noise_ratio(clean, other, data_range=255)


def calculate_ssim(clean_path, other_path):
    clean = cv2.imread(clean_path)
    other = cv2.imread(other_path)

    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    other_gray = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)

    return structural_similarity(clean_gray, other_gray, data_range=255)


if __name__ == "__main__":
    clean_path = "data/clean/car1.jpg"
    other_path = "data/clean/car1.jpg"

    psnr = calculate_psnr(clean_path, other_path)
    ssim = calculate_ssim(clean_path, other_path)

    print("PSNR:", psnr)
    print("SSIM:", ssim)