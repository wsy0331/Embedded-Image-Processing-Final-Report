import cv2
import numpy as np
import time

def detect_fire(curr_frame, bg_subtractor, kernel, lower_fire, upper_fire, display_frame):
    motion_mask = bg_subtractor.apply(curr_frame)
    _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    red_orange_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
    fire_mask = cv2.bitwise_and(motion_mask, red_orange_mask)
    
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 15:
            cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
    return fire_mask

def detect_smoke_motion(fx, fy, step, min_speed, display_frame, fire_mask):
    """只負責計算 FoE 與繪圖，不計算光流"""
    t0 = time.perf_counter()
    height, width = fx.shape
    y, x = np.mgrid[step//2:height:step, step//2:width:step].reshape(2, -1).astype(int)
    
    fx_sampled, fy_sampled = fx[y, x], fy[y, x]
    magnitude = np.hypot(fx_sampled, fy_sampled)
    
    valid_idx = (magnitude > min_speed) & (fire_mask[y, x] == 0)
    valid_x, valid_y = x[valid_idx], y[valid_idx]
    
    # FoE 求解
    foe_point = None
    if len(valid_x) >= 2:
        A = np.vstack((fy_sampled[valid_idx], -fx_sampled[valid_idx])).T
        b = (fy_sampled[valid_idx] * valid_x - fx_sampled[valid_idx] * valid_y).reshape(-1, 1)
        try:
            foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            foe_point = (int(foe[0][0]), int(foe[1][0]))
        except: pass
    
    t_foe_calc = (time.perf_counter() - t0) * 1000
    
    # 繪圖
    t0 = time.perf_counter()
    for i in range(len(valid_x)):
        cv2.arrowedLine(display_frame, (valid_x[i], valid_y[i]), 
                        (int(valid_x[i] + fx_sampled[valid_idx][i]*2), int(valid_y[i] + fy_sampled[valid_idx][i]*2)), 
                        (255, 0, 0), 1, cv2.LINE_AA)
    if foe_point: cv2.circle(display_frame, foe_point, 6, (0, 0, 255), -1)
    
    return t_foe_calc, (time.perf_counter() - t0) * 1000