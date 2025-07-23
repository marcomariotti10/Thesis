import os
import numpy as np
import matplotlib.image as mpimg
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

def load_png_as_array(path):
    img = mpimg.imread(path)
    # If image is RGB or RGBA, convert to grayscale
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)

def compare_images(img1, img2):
    v1 = img1.flatten()
    v2 = img2.flatten()

    mae = np.mean(np.abs(v1 - v2))
    ssd = np.sum((v1 - v2) ** 2)
    cosine = cosine_similarity([v1], [v2])[0][0]
    euclidean = norm(v1 - v2)
    correlation = np.corrcoef(v1, v2)[0, 1]

    return {
        "MAE": mae,
        "SSD": ssd,
        "Cosine Similarity": cosine,
        "Euclidean Distance": euclidean,
        "Pearson Correlation": correlation
    }

def load_and_compare(base_path, fold_index):
    train_path = os.path.join(base_path, f"train_eigenimage_1_fold_{fold_index}.png")
    val_path = os.path.join(base_path, f"val_eigenimage_1_fold_{fold_index}.png")
    test_path = os.path.join(base_path, f"test_eigenimage_1_fold_{fold_index}.png")

    train_img = load_png_as_array(train_path)
    val_img = load_png_as_array(val_path)
    test_img = load_png_as_array(test_path)

    print(f"\n--- Eigenimage Differences for Fold {fold_index} ---\n")

    print("Train vs Val:")
    print(compare_images(train_img, val_img))

    print("\nTrain vs Test:")
    print(compare_images(train_img, test_img))

    print("\nVal vs Test:")
    print(compare_images(val_img, test_img))

if __name__ == '__main__':
    base_path = '.'  # or wherever your PNGs are saved
    fold_index = 0   # Adjust if you're analyzing a different fold
    load_and_compare(base_path, fold_index)
