import cv2

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")

ruby_crop = img[135:175, 550:625]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\real_ruby.png", ruby_crop)

sapphire_crop = img[440:485, 580:685]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\real_sapphire.png", sapphire_crop)

print("Saved real_ruby.png and real_sapphire.png")
