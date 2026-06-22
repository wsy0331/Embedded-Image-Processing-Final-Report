import os
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 全域變數與字體載入
# ==========================================
TARGET_FPS = 15

def on_fps_change(val):
    global TARGET_FPS
    TARGET_FPS = val if val > 0 else 1

def get_chinese_font(size):
    """跨平台載入中文字體"""
    try:
        return ImageFont.truetype("msjh.ttc", size)  # Windows 微軟正黑體
    except IOError:
        try:
            return ImageFont.truetype("PingFang.ttc", size)  # Mac 蘋方體
        except IOError:
            return ImageFont.load_default()  # Fallback

# ==========================================
# 效能監控儀表板 (Gantt Chart)
# ==========================================
def draw_timeline_dashboard(frame_times, target_fps):
    canvas_h, canvas_w = 550, 800
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    deadline_ms = 1000.0 / target_fps
    px_per_ms = 3.0
    text_x_offset = 10
    bar_start_x = 180 
    
    total_time = sum(frame_times.values())
    status_color_bgr = (0, 255, 0) if total_time <= deadline_ms else (0, 0, 255)
    status_color_rgb = (status_color_bgr[2], status_color_bgr[1], status_color_bgr[0])

    # 1. 繪製圖形框線
    y_offset = 60
    current_time_ms = 0.0
    for name, duration in frame_times.items():
        start_px = bar_start_x + int(current_time_ms * px_per_ms)
        width_px = max(2, int(duration * px_per_ms))
        end_px = start_px + width_px
        
        cv2.rectangle(canvas, (start_px, y_offset), (end_px, y_offset + 25), (180, 180, 240), -1)
        cv2.rectangle(canvas, (start_px, y_offset), (end_px, y_offset + 25), (100, 100, 200), 1)
        
        current_time_ms += duration
        y_offset += 35
        
    deadline_px = bar_start_x + int(deadline_ms * px_per_ms)
    cv2.line(canvas, (deadline_px, 50), (deadline_px, canvas_h - 20), (0, 0, 255), 2)

    # 2. 繪製中文字與精確數據
    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_title = get_chinese_font(22)
    font_label = get_chinese_font(16)
    font_small = get_chinese_font(14)

    draw.text((10, 20), f"Target: {target_fps} FPS (Deadline: {deadline_ms:.3f} ms)", font=font_title, fill=(255, 255, 255))
    draw.text((450, 20), f"Total Time: {total_time:.3f} ms", font=font_title, fill=status_color_rgb)
    draw.text((deadline_px + 5, 45), "Deadline", font=font_small, fill=(255, 0, 0))

    y_offset = 60
    current_time_ms = 0.0
    for name, duration in frame_times.items():
        start_px = bar_start_x + int(current_time_ms * px_per_ms)
        width_px = max(2, int(duration * px_per_ms))
        end_px = start_px + width_px
        
        draw.text((text_x_offset, y_offset + 2), name, font=font_label, fill=(200, 200, 200))
        draw.text((end_px + 5, y_offset + 4), f"{duration:.3f}ms", font=font_small, fill=(255, 255, 0))
        
        current_time_ms += duration
        y_offset += 35

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 影像處理核心演算法
# ==========================================
def detect_fire(curr_frame, bg_subtractor, kernel, lower_fire, upper_fire, display_frame):
    t0 = time.perf_counter()
    motion_mask = bg_subtractor.apply(curr_frame)
    _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    t_mog2 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    red_orange_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
    lower_white, upper_white = np.array([0, 0, 220]), np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    combined_color_mask = cv2.bitwise_or(red_orange_mask, white_mask)
    fire_mask = cv2.bitwise_and(motion_mask, combined_color_mask)
    t_hsv = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t_contours = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for contour in fire_contours:
        if cv2.contourArea(contour) > 15:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
    t_draw = (time.perf_counter() - t0) * 1000

    return fire_mask, t_mog2, t_hsv, t_contours, t_draw


def detect_smoke(old_gray, new_gray, step, min_speed, display_frame, fire_mask):
    # 1. Farneback 光流法
    t0 = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    t_flow = (time.perf_counter() - t0) * 1000

    # 2. 光流法輸出的高斯模糊平滑
    t0 = time.perf_counter()
    fx, fy = cv2.split(flow)
    fx = cv2.GaussianBlur(fx.astype(np.float32), (5, 5), 0)
    fy = cv2.GaussianBlur(fy.astype(np.float32), (5, 5), 0)
    t_flow_blur = (time.perf_counter() - t0) * 1000

    # 3. 矩陣取樣與 FoE 計算 (最小平方法)
    t0 = time.perf_counter()
    height, width = new_gray.shape
    y, x = np.mgrid[step//2:height:step, step//2:width:step].reshape(2, -1).astype(int)
    fx_sampled, fy_sampled = fx[y, x], fy[y, x]
    magnitude = np.hypot(fx_sampled, fy_sampled)
    valid_idx = (magnitude > min_speed) & (fire_mask[y, x] == 0)

    valid_x, valid_y = [], []
    valid_fx, valid_fy = [], []
    foe_point = None

    if not (np.var(fx_sampled[valid_idx]) < 0.5 and np.var(fy_sampled[valid_idx]) < 0.5):
        valid_x, valid_y = x[valid_idx], y[valid_idx]
        valid_fx, valid_fy = fx_sampled[valid_idx], fy_sampled[valid_idx]

        if len(valid_x) >= 2:
            A = np.vstack((valid_fy, -valid_fx)).T
            b = (valid_fy * valid_x - valid_fx * valid_y).reshape(-1, 1)
            try:
                foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                foe_x, foe_y = int(foe[0][0]), int(foe[1][0])
                if 0 <= foe_x < width and 0 <= foe_y < height:
                    foe_point = (foe_x, foe_y)
            except Exception:
                pass
    t_foe_calc = (time.perf_counter() - t0) * 1000

    # 4. 標記煙霧 (將畫藍色箭頭與畫紅色 FoE 圓點整併在一起計時)
    t0 = time.perf_counter()
    for i in range(len(valid_x)):
        end_x = int(valid_x[i] + valid_fx[i] * 2)
        end_y = int(valid_y[i] + valid_fy[i] * 2)
        if 0 <= end_x < width and 0 <= end_y < height and fire_mask[end_y, end_x] == 0:
            cv2.arrowedLine(display_frame, (valid_x[i], valid_y[i]), (end_x, end_y), (255, 0, 0), 1, tipLength=0.5)
            
    if foe_point is not None:
        cv2.circle(display_frame, foe_point, 6, (0, 0, 255), -1)
    t_draw_smoke = (time.perf_counter() - t0) * 1000
    
    return t_flow, t_flow_blur, t_foe_calc, t_draw_smoke

# ==========================================
# 主程式
# ==========================================
def main():
    print("\n--- 程式開始執行 ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "fire_smoke6.mp4")
    
    print(f"🔍 系統目前正在此位置尋找影片：\n👉 {video_path}")

    if not os.path.exists(video_path):
        print("\n❌ 錯誤：找不到影片檔案！請確認影片是否放在正確的資料夾。")
        return
    else:
        print("✅ 作業系統確認影片檔案存在。")

    input_video = cv2.VideoCapture(video_path)
    if not input_video.isOpened():
        print("\n❌ 錯誤：OpenCV 找到了檔案，但無法開啟！可能是解碼器問題。")
        return

    ret, first_frame = input_video.read()
    if not ret: 
        print("\n❌ 錯誤：無法讀取第一幀畫面！")
        return
    else:
        print("✅ 成功讀取畫面！啟動儀表板視窗中...\n")

    cv2.namedWindow("Performance Timeline")
    cv2.createTrackbar("Target FPS", "Performance Timeline", 15, 60, on_fps_change)
    
    target_size = (640, 480)
    old_gray = cv2.cvtColor(cv2.resize(first_frame, target_size), cv2.COLOR_BGR2GRAY)
    old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

    bg_subtractor_fire = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fire_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lower_fire, upper_fire = np.array([0, 100, 150]), np.array([60, 255, 255])

    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret: 
            input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        new_frame = cv2.resize(frame, target_size)
        
        current_frame_times = {}

        # --- 基礎影像前處理 ---
        t0 = time.perf_counter()
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        current_frame_times["灰階"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        new_gauss = cv2.GaussianBlur(new_gray, (5, 5), 0)
        current_frame_times["高斯模糊 (影像)"] = (time.perf_counter() - t0) * 1000

        # --- 火災偵測分支 ---
        fire_mask, t_mog2, t_hsv, t_contours, t_draw_fire = detect_fire(
            new_frame, bg_subtractor_fire, fire_kernel, lower_fire, upper_fire, new_frame
        )
        current_frame_times["MOG2 運動偵測"] = t_mog2
        current_frame_times["HSV"] = t_hsv
        current_frame_times["Contours 輪廓"] = t_contours
        current_frame_times["標記火焰"] = t_draw_fire

        # --- 煙霧偵測分支 ---
        fire_mask_dilated = cv2.dilate(fire_mask, dilate_kernel)
        t_flow, t_flow_blur, t_foe_calc, t_draw_smoke = detect_smoke(
            old_gray, new_gauss, 10, 0.7, new_frame, fire_mask_dilated
        )
        current_frame_times["Farneback 光流法"] = t_flow
        current_frame_times["高斯模糊 (光流)"] = t_flow_blur
        current_frame_times["FoE 計算"] = t_foe_calc
        current_frame_times["標記煙霧"] = t_draw_smoke  # 🌟 FoE 與藍色箭頭繪製時間合併於此

        # --- HUD 與 畫面輸出 ---
        deadline_ms = 1000.0 / TARGET_FPS
        total_time_ms = sum(current_frame_times.values())
        status_color = (0, 255, 0) if total_time_ms <= deadline_ms else (0, 0, 255)
        status_text = "OK" if total_time_ms <= deadline_ms else "OVER BUDGET"

        cv2.putText(new_frame, f"Target: {TARGET_FPS} FPS | Deadline: {deadline_ms:.3f} ms", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(new_frame, f"Total Time: {total_time_ms:.3f} ms [{status_text}]", 
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        dashboard_img = draw_timeline_dashboard(current_frame_times, TARGET_FPS)
        cv2.imshow("Performance Timeline", dashboard_img)
        cv2.imshow("Output Video", new_frame)
        
        old_gray = new_gauss.copy()
        if cv2.waitKey(25) >= 0:
            break

    input_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()