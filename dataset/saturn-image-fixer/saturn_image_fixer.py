import cv2
import numpy as np
import glob

# Process multiple images in a folder
for img_path in glob.glob("images/*.jpg"):
    img = cv2.imread(img_path)
    height, width, _ = img.shape # Get image dimensions

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create masks for background and shadow
    mask = cv2.inRange(img, (170, 170, 170), (255, 255, 255)) # Detect background and shadow

    # Apply black color where mask is detected
    img[mask > 0] = [0, 0, 0]

    # Remove remaining grey pixels in background (the background had darker grey pixels in specific areas)
    img[height - 1, :] = [0, 0, 0] # Fill the bottom row with black
    img[:1, :1] = [0, 0, 0] # Fill top-left pixel with black

    # Save the processed image
    cv2.imwrite(img_path, img)

print("Processing complete!")
