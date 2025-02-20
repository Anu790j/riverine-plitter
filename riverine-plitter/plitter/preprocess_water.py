import cv2
import numpy as np

def preprocess_water_image(image):
    """
    Preprocess the input image to enhance features in water scenes.
    Steps:
      1. Convert image to HSV.
      2. Apply CLAHE to the V channel.
      3. Boost saturation.
      4. Convert back to BGR.
      5. Apply median blur for fast denoising.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply CLAHE to the V channel for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)
    
    # Boost saturation by 20%
    s = cv2.multiply(s, np.array([1.2]))
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge channels and convert back to BGR
    hsv_enhanced = cv2.merge([h, s, v_clahe])
    enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # Use a median blur for denoising (kernel size 5)
    denoised = cv2.medianBlur(enhanced_img, 5)
    
    return denoised
