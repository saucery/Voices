import cv2
import numpy as np

img = cv2.imread(r"C:\Users\gregg\.gemini\antigravity-ide\brain\bc444d8e-d778-41d6-a9f9-4d86757c5719\gems_screenshot.png")

ruby_crop = img[135:175, 550:625]
sapphire_crop = img[440:485, 580:685]
belt_crop = img[590:635, 320:550]

for name, crop in [("RUBY", ruby_crop), ("SAPPHIRE", sapphire_crop), ("DOUBLE BELT", belt_crop)]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(crop)
    print(f"\n==================== {name} ====================")
    # Find text pixels by brightness (V > 140)
    text_pts = np.where(hsv[:,:,2] > 140)
    print(f"Total bright text pixels: {len(text_pts[0])}")
    if len(text_pts[0]) > 0:
        h_vals = hsv[:,:,0][text_pts]
        s_vals = hsv[:,:,1][text_pts]
        v_vals = hsv[:,:,2][text_pts]
        r_vals = r[text_pts]
        g_vals = g[text_pts]
        b_vals = b[text_pts]
        print(f"Hue: min={h_vals.min()}, median={np.median(h_vals):.1f}, max={h_vals.max()}")
        print(f"Sat: min={s_vals.min()}, median={np.median(s_vals):.1f}, max={s_vals.max()}")
        print(f"Val: min={v_vals.min()}, median={np.median(v_vals):.1f}, max={v_vals.max()}")
        print(f"B: median={np.median(b_vals):.1f}, max={b_vals.max()}")
        print(f"G: median={np.median(g_vals):.1f}, max={g_vals.max()}")
        print(f"R: median={np.median(r_vals):.1f}, max={r_vals.max()}")
        # G / R ratio
        gr_ratio = g_vals.astype(float) / np.maximum(1.0, r_vals.astype(float))
        print(f"G/R ratio: median={np.median(gr_ratio):.2f}, min={gr_ratio.min():.2f}, max={gr_ratio.max():.2f}")

    # Find background pixels (V in [20, 120])
    bg_pts = np.where((hsv[:,:,2] >= 20) & (hsv[:,:,2] <= 120))
    print(f"Total BG pixels: {len(bg_pts[0])}")
    if len(bg_pts[0]) > 0:
        bg_h = hsv[:,:,0][bg_pts]
        bg_s = hsv[:,:,1][bg_pts]
        bg_v = hsv[:,:,2][bg_pts]
        bg_r = r[bg_pts]
        bg_g = g[bg_pts]
        bg_b = b[bg_pts]
        print(f"BG Hue: min={bg_h.min()}, median={np.median(bg_h):.1f}, max={bg_h.max()}")
        print(f"BG Sat: min={bg_s.min()}, median={np.median(bg_s):.1f}, max={bg_s.max()}")
        print(f"BG Val: min={bg_v.min()}, median={np.median(bg_v):.1f}, max={bg_v.max()}")
        print(f"BG BGR: ({np.median(bg_b):.1f}, {np.median(bg_g):.1f}, {np.median(bg_r):.1f})")
        # In olive BG: G is significant (G >= 35, G >= B * 1.3)
        # In black BG (Double belt): B, G, R are all ~ 0-15
        olive_pts = np.count_nonzero((bg_g >= 30) & (bg_g >= bg_b + 5))
        print(f"Olive BG pixels (G>=30, G>=B+5): {olive_pts}/{len(bg_pts[0])} ({olive_pts/len(bg_pts[0])*100:.1f}%)")
