import cv2
import numpy as np

def load_image(path, flag=cv2.IMREAD_COLOR):
    image = cv2.imread(path, flag)
    if image is None:
        raise FileNotFoundError(f"Image at path '{path}' could not be loaded.")
    return image

def apply_displacement_map(pattern, flag_gray, strength=0.3):
    """
    Warps the pattern using wave-like folds extracted from the white flag image.

    """
    h, w = pattern.shape[:2]

    # Calculate gradients (Sobel) to simulate flag folds
    grad_x = cv2.Sobel(flag_gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(flag_gray, cv2.CV_32F, 0, 1, ksize=5)

    dx = cv2.GaussianBlur(grad_x, (0, 0), 3)
    dy = cv2.GaussianBlur(grad_y, (0, 0), 3)

    dx = cv2.normalize(dx, None, -strength, strength, cv2.NORM_MINMAX)
    dy = cv2.normalize(dy, None, -strength, strength, cv2.NORM_MINMAX)

    # Create remapping grid
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(map_x + dx * w)
    map_y = np.float32(map_y + dy * h)

    warped = cv2.remap(pattern, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def blend_images(base_img, overlay_img, alpha_mask):
    """
    Blends the overlay image onto the base using alpha blending.
    """
    overlay = overlay_img.astype(float)
    base = base_img.astype(float)
    alpha = alpha_mask.astype(float) / 255.0

    if len(alpha.shape) == 2:
        alpha = cv2.merge([alpha, alpha, alpha])

    blended = cv2.convertScaleAbs(base * (1 - alpha) + overlay * alpha)
    return blended

def main():
    try:
        # Load user-provided images
        flag = load_image("Flag.png")
        pattern = load_image("Pattern.png")

        # Resize pattern to match flag dimensions
        pattern_resized = cv2.resize(pattern,  (flag.shape[1], flag.shape[0]))

        # Convert flag to grayscale to extract folds
        flag_gray = cv2.cvtColor(flag, cv2.COLOR_BGR2GRAY)

        # Warp pattern to follow flag folds
        warped_pattern = apply_displacement_map(pattern_resized, flag_gray, strength=0.35)

        # Use enhanced brightness/contrast from flag to create alpha mask
        contrast = cv2.equalizeHist(flag_gray)
        alpha_mask = cv2.GaussianBlur(contrast, (0, 0), 7)
        alpha_mask = cv2.normalize(alpha_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Blend images
        output = blend_images(flag, warped_pattern, alpha_mask)

        # Save final result
        cv2.imwrite("Output.jpg", output)
        print("✅ Output.jpg saved successfully.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
