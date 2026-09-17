import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Let's find Ruby and Sapphire coordinates in this image (716x819)
# Ruby is around y: 175-245, x: 660-760
# Sapphire is around y: 620-690, x: 700-810

ruby_crop = img[180:240, 670:760]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\ruby_crop.png", ruby_crop)

sapphire_crop = img[630:690, 700:810]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\sapphire_crop.png", sapphire_crop)

print("Saved ruby_crop.png and sapphire_crop.png")
