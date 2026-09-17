import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")

# Exact coordinates
ruby_box = img[184:236, 672:754]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\exact_ruby.png", ruby_box)

sapphire_box = img[634:684, 706:838]
cv2.imwrite(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\exact_sapphire.png", sapphire_box)

print(f"Exact Ruby shape: {ruby_box.shape}")
print(f"Exact Sapphire shape: {sapphire_box.shape}")

# Inspect color characteristics of Ruby & Sapphire
for name, box in [("RUBY", ruby_box), ("SAPPHIRE", sapphire_box)]:
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(box)
    print(f"\n--- {name} Color Profile ---")
    print(f"BGR min/mean/max:")
    print(f"  B: {b.min()} / {b.mean():.1f} / {b.max()}")
    print(f"  G: {g.min()} / {g.mean():.1f} / {g.max()}")
    print(f"  R: {r.min()} / {r.mean():.1f} / {r.max()}")
    print(f"HSV min/mean/max:")
    print(f"  H: {hsv[:,:,0].min()} / {hsv[:,:,0].mean():.1f} / {hsv[:,:,0].max()}")
    print(f"  S: {hsv[:,:,1].min()} / {hsv[:,:,1].mean():.1f} / {hsv[:,:,1].max()}")
    print(f"  V: {hsv[:,:,2].min()} / {hsv[:,:,2].mean():.1f} / {hsv[:,:,2].max()}")

    # Border inspection (outer 2 pixels)
    border_mask = np.zeros(box.shape[:2], dtype=bool)
    border_mask[:2, :] = True
    border_mask[-2:, :] = True
    border_mask[:, :2] = True
    border_mask[:, -2:] = True
    
    border_b = b[border_mask]
    border_g = g[border_mask]
    border_r = r[border_mask]
    border_h = hsv[:,:,0][border_mask]
    border_s = hsv[:,:,1][border_mask]
    border_v = hsv[:,:,2][border_mask]
    print(f"Border (outer 2px) BGR mean: ({border_b.mean():.1f}, {border_g.mean():.1f}, {border_r.mean():.1f})")
    print(f"Border HSV mean: H={border_h.mean():.1f}, S={border_s.mean():.1f}, V={border_v.mean():.1f}")
    
    # Inside background vs text
    inner_mask = ~border_mask
    print(f"Inside BGR mean: ({b[inner_mask].mean():.1f}, {g[inner_mask].mean():.1f}, {r[inner_mask].mean():.1f})")
