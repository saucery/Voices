import cv2
import os
import shutil

src_path = r"C:\Users\gregg\OneDrive\Pictures\Screenshots\Screenshot 2026-09-17 110007.png"
dst_path = r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png"

if os.path.exists(src_path):
    shutil.copyfile(src_path, dst_path)
    print(f"Copied successfully to {dst_path}")
    img = cv2.imread(src_path)
    print(f"Image shape: {img.shape}")
else:
    print(f"Source file does not exist: {src_path}")
