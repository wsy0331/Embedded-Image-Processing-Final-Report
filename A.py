import os
import cv2
import numpy as np
import time
import multiprocessing as mp
import psutil
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# Global Variables
# ==========================================
TARGET_FPS = 15

def get_english_font():
    return ImageFont.load_default()

# ==========================================
# Performance Dashboard
# ==========================================
def draw_timeline_dashboard(frame_times, target_fps):
    canvas_h, canvas_w = 550, 800
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    deadline_ms = 1000.0 / target_fps
    px_per_ms = 3.0
    text_x_offset = 10
    bar_start_x = 220 
    
    total_time = sum(frame_times.values())
    status_color_bgr = (0, 255, 0) if total_time <= deadline_ms else (0, 0, 255)
    status_color_rgb = (status_color_bgr[2], status_color_bgr[1], status_color_bgr[0])

    y_offset = 60
    current_time_ms = 0.0
    for name, duration in frame_times.items():
        start_px = bar_start_x + int(current_time_ms * px_per_ms)
        width_px = max(2, int(duration * px_per_ms))
        end_px = start_px + width_px
        
        if "[C0]" in name:
            box_color, edge_color = (204, 229, 255), (102, 178, 255) 
        elif "[C1]" in name:
            box_color, edge_color = (229, 204, 255), (178, 102, 255) 
        else:
            box_color, edge_color = (204, 255, 255), (102, 255, 255) 

        cv2.rectangle(canvas, (start_px, y_offset), (end_px, y_offset + 25), box_color, -1)
        cv2.rectangle(canvas, (start_px, y_offset), (end_px, y_offset + 25), edge_color, 1)
        
        current_time_ms += duration
        y_offset += 35

    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_default = get_english_font()

    draw.text((10, 20), f"Target: {target_fps} FPS", font=font_default, fill=(255, 255, 255))
    draw.text((450, 20), f"Pipeline Latency: {total_time:.3f} ms", font=font_default, fill=status_color_rgb)

    y_offset = 60
    current_time_ms = 0.0
    for name, duration in frame_times.items():
        start_px = bar_start_x + int(current_time_ms * px_per_ms)
        width_px = max(2, int(duration * px_per_ms))
        end_px = start_px + width_px
        
        draw.text((text_x_offset, y_offset + 5), name, font=font_default, fill=(200, 200, 200))
        draw.text((end_px + 5, y_offset + 5), f"{duration:.3f}ms", font=font_default, fill=(255, 255, 0))
        current_time_ms += duration
        y_offset += 35

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# Core Functions
# ==========================================
def execute_core0(curr_frame, bg_subtractor, fire_kernel, lower_fire, upper_fire):
    times = {}
    display_frame = curr_frame.copy()
    
    t0 = time.perf_counter()
    new_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    times["[C0] Grayscale"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    small_gray = cv2.resize(new_gray, (0, 0), fx=0.5, fy=0.5)
    new_gauss_small = cv2.GaussianBlur(small_gray, (5, 5), 0)
    times["[C0] Gauss Blur (Half)"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    motion_mask = bg_subtractor.apply(curr_frame)
    _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, fire_kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, fire_kernel)
    times["[C0] MOG2 Motion"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    red_orange_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
    lower_white, upper_white = np.array([0, 0, 220]), np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    combined_color_mask = cv2.bitwise_or(red_orange_mask, white_mask)
    fire_mask = cv2.bitwise_and(motion_mask, combined_color_mask)
    times["[C0] HSV Color"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    times["[C0] Contours"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for contour in fire_contours:
        if cv2.contourArea(contour) > 15:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
    times["[C0] Draw Fire"] = (time.perf_counter() - t0) * 1000

    return new_gauss_small, fire_mask, display_frame, times

def execute_core1(old_gray_gauss_small, new_gray_gauss_small):
    times = {}
    t0 = time.perf_counter()
    flow_small = cv2.calcOpticalFlowFarneback(old_gray_gauss_small, new_gray_gauss_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    times["[C1] Farneback"] = (time.perf_counter() - t0) * 1000
    flow = cv2.resize(flow_small, (0, 0), fx=2.0, fy=2.0) * 2.0
    return flow, times

def execute_core2(flow, fire_mask, display_frame, step, min_speed, dilate_kernel):
    times = {}
    
    t0 = time.perf_counter()
    fx, fy = cv2.split(flow)
    fx = cv2.GaussianBlur(fx.astype(np.float32), (5, 5), 0)
    fy = cv2.GaussianBlur(fy.astype(np.float32), (5, 5), 0)
    times["[C2] Flow Blur"] = (time.perf_counter() - t0) * 1000

    fire_mask_dilated = cv2.dilate(fire_mask, dilate_kernel)

    t0 = time.perf_counter()
    height, width = fire_mask.shape
    y, x = np.mgrid[step//2:height:step, step//2:width:step].reshape(2, -1).astype(int)
    fx_sampled, fy_sampled = fx[y, x], fy[y, x]
    magnitude = np.hypot(fx_sampled, fy_sampled)
    
    valid_idx = (magnitude > min_speed) & (fire_mask_dilated[y, x] == 0)

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
    times["[C2] FoE Calc"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for i in range(len(valid_x)):
        end_x = int(valid_x[i] + valid_fx[i] * 2)
        end_y = int(valid_y[i] + valid_fy[i] * 2)
        if 0 <= end_x < width and 0 <= end_y < height and fire_mask_dilated[end_y, end_x] == 0:
            cv2.arrowedLine(display_frame, (valid_x[i], valid_y[i]), (end_x, end_y), (255, 0, 0), 1, tipLength=0.5)
            
    if foe_point is not None:
        cv2.circle(display_frame, foe_point, 6, (0, 0, 255), -1)
    times["[C2] Draw Smoke"] = (time.perf_counter() - t0) * 1000
    
    return display_frame, times

# ==========================================
# True Multiprocessing Workers with CPU Affinity
# ==========================================
def worker_core0(q_in, q_out_c1, q_out_c2):
    # Bind to Physical CPU Core 1
    psutil.Process(os.getpid()).cpu_affinity([1])
    # Prevent OpenCV internal multi-threading from fighting our affinity
    cv2.setNumThreads(1) 

    bg_subtractor_fire = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    fire_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lower_fire, upper_fire = np.array([0, 100, 150]), np.array([60, 255, 255])

    while True:
        curr_frame = q_in.get()
        if curr_frame is None:
            q_out_c1.put(None)
            q_out_c2.put(None)
            break

        new_gauss_small, fire_mask, display_frame, times = execute_core0(
            curr_frame, bg_subtractor_fire, fire_kernel, lower_fire, upper_fire
        )
        
        q_out_c1.put(new_gauss_small)
        q_out_c2.put((fire_mask, display_frame, times))

def worker_core1(q_in, q_out):
    # Bind to Physical CPU Core 2
    psutil.Process(os.getpid()).cpu_affinity([2])
    cv2.setNumThreads(1)
    
    old_gray_gauss_small = None

    while True:
        new_gray_gauss_small = q_in.get()
        if new_gray_gauss_small is None:
            q_out.put(None)
            break

        if old_gray_gauss_small is None:
            old_gray_gauss_small = new_gray_gauss_small
            dummy_flow = np.zeros((new_gray_gauss_small.shape[0], new_gray_gauss_small.shape[1], 2), dtype=np.float32)
            dummy_flow = cv2.resize(dummy_flow, (0, 0), fx=2.0, fy=2.0)
            q_out.put((dummy_flow, {"[C1] Farneback": 0.0}))
            continue

        flow, times = execute_core1(old_gray_gauss_small, new_gray_gauss_small)
        q_out.put((flow, times))
        old_gray_gauss_small = new_gray_gauss_small

def worker_core2(q_in_c0, q_in_c1, q_out):
    # Bind to Physical CPU Core 3
    psutil.Process(os.getpid()).cpu_affinity([3])
    cv2.setNumThreads(1)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    while True:
        data_c0 = q_in_c0.get()
        data_c1 = q_in_c1.get()
        
        if data_c0 is None or data_c1 is None:
            q_out.put(None)
            break
            
        fire_mask, display_frame, times_c0 = data_c0
        flow, times_c1 = data_c1

        final_frame, times_c2 = execute_core2(flow, fire_mask, display_frame, 10, 0.7, dilate_kernel)
        
        # Merge all times
        all_times = {**times_c0, **times_c1, **times_c2}
        q_out.put((final_frame, all_times))

# ==========================================
# Main Execution (CPU 0)
# ==========================================
def main():
    # Bind Main Process to Physical CPU Core 0
    psutil.Process(os.getpid()).cpu_affinity([0])

    print("\n--- Physical Multi-Processing Pipeline Started ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "fire_smoke2.mp4")
    
    if not os.path.exists(video_path):
        print(f"\n[ERROR] Video file not found: {video_path}")
        return

    input_video = cv2.VideoCapture(video_path)
    if not input_video.isOpened():
        print("\n[ERROR] OpenCV cannot open video.")
        return

    # Create Multiprocessing Queues
    q_frame_in = mp.Queue(maxsize=3)
    q_c0_to_c1 = mp.Queue(maxsize=3)
    q_c0_to_c2 = mp.Queue(maxsize=3)
    q_c1_to_c2 = mp.Queue(maxsize=3)
    q_final_out = mp.Queue(maxsize=3)

    # Initialize and start workers
    p0 = mp.Process(target=worker_core0, args=(q_frame_in, q_c0_to_c1, q_c0_to_c2))
    p1 = mp.Process(target=worker_core1, args=(q_c0_to_c1, q_c1_to_c2))
    p2 = mp.Process(target=worker_core2, args=(q_c0_to_c2, q_c1_to_c2, q_final_out))

    p0.start()
    p1.start()
    p2.start()

    cv2.namedWindow("Performance Timeline")
    
    target_size = (640, 480)

    try:
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret: 
                input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            new_frame = cv2.resize(frame, target_size)
            
            # Feed frame to the pipeline
            q_frame_in.put(new_frame)

            # Receive processed frame and stats from the end of pipeline
            result = q_final_out.get()
            if result is None:
                break
                
            final_frame, all_times = result

            # Draw UI
            total_time_ms = sum(all_times.values())
            deadline_ms = 1000.0 / TARGET_FPS
            status_color = (0, 255, 0) if total_time_ms <= deadline_ms else (0, 0, 255)

            cv2.putText(final_frame, f"Target: {TARGET_FPS} FPS", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(final_frame, f"Latency: {total_time_ms:.1f} ms", 
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            dashboard_img = draw_timeline_dashboard(all_times, TARGET_FPS)
            cv2.imshow("Performance Timeline", dashboard_img)
            cv2.imshow("Output Video", final_frame)
            
            if cv2.waitKey(1) >= 0:
                break

    finally:
        # Poison pills for clean shutdown
        q_frame_in.put(None)
        p0.join()
        p1.join()
        p2.join()
        input_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()