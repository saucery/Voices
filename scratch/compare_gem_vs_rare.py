import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")

ruby_crop = img[135:175, 550:625]
sapphire_crop = img[440:485, 580:685]
double_belt_crop = img[590:635, 320:550]
wand_crop = img[180:225, 500:770]

def analyze_crop(name, crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(crop)
    print(f"\n=== {name} ===")
    
    # Yellow/Lime text mask: H in [20, 38], S in [120, 255], V > 160
    # Lime/Chartreuse Hue is around 25-35 in OpenCV HSV
    lime_mask = (hsv[:,:,0] >= 22) & (hsv[:,:,0] <= 38) & (hsv[:,:,1] >= 100) & (hsv[:,:,2] >= 160)
    
    # Gold/Orange text mask (Tier 2 rares): H in [10, 21], S in [150, 255], V > 160
    gold_mask = (hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 21) & (hsv[:,:,1] >= 150) & (hsv[:,:,2] >= 160)
    
    # Dark olive background mask: H in [20, 45], S in [40, 150], V in [20, 100]
    olive_bg_mask = (hsv[:,:,0] >= 20) & (hsv[:,:,0] <= 45) & (hsv[:,:,1] >= 30) & (hsv[:,:,2] >= 20) & (hsv[:,:,2] <= 110)
    
    print(f"Size: {crop.shape}")
    print(f"Lime pixels (Gem text/border): {np.count_nonzero(lime_mask)} ({np.mean(lime_mask)*100:.1f}%)")
    print(f"Gold/Orange pixels (Rare item text): {np.count_nonzero(gold_mask)} ({np.mean(gold_mask)*100:.1f}%)")
    print(f"Olive BG pixels: {np.count_nonzero(olive_bg_mask)} ({np.mean(olive_bg_mask)*100:.1f}%)")
    
    # Text color sample: top 5% brightest pixels
    brightest = crop[hsv[:,:,2] > 180]
    if len(brightest) > 0:
        b_mean = brightest[:, 0].mean()
        g_mean = brightest[:, 1].mean()
        r_mean = brightest[:, 2].mean()
        print(f"Bright text BGR: ({b_mean:.1f}, {g_mean:.1f}, {r_mean:.1f}) -> G/R ratio: {g_mean/max(1.0, r_mean):.2f}, B: {b_mean:.1f}")

analyze_crop("RUBY", ruby_crop)
analyze_crop("SAPPHIRE", sapphire_crop)
analyze_crop("DOUBLE BELT (TIER 2)", double_belt_crop)
analyze_crop("GALVANIC WAND (TIER 2)", wand_crop)
