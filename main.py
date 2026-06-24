import os, cv2, time, threading, queue
import numpy as np
from vision_modules import detect_fire, detect_smoke_motion

# 1. 確保參數定義在最上方
IMAGE_SIZE = (640, 480)
ENABLE_SMOKE_ANALYSIS = True

data_lock = threading.Lock()
current_raw_frame = None
latest_fire_canvas = latest_smoke_canvas = None
t_fire_branch = t_core2 = t_core3 = 0.0
flow_queue = queue.Queue(maxsize=2)

def fire_worker(target_size):
    global latest_fire_canvas, t_fire_branch
    bg = cv2.createBackgroundSubtractorMOG2(history=500)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while True:
        with data_lock: frame = current_raw_frame.copy() if current_raw_frame is not None else None
        if frame is None: continue
        canvas = frame.copy()
        t0 = time.perf_counter()
        detect_fire(frame, bg, k, np.array([0,100,150]), np.array([60,255,255]), canvas)
        with data_lock: latest_fire_canvas, t_fire_branch = canvas, (time.perf_counter()-t0)*1000

def optical_flow_worker(video_path, target_size, dilate_kernel):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    old_gray = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2GRAY)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        new_gray = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2GRAY)
        
        t0 = time.perf_counter()
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = cv2.split(flow)
        t_flow = (time.perf_counter() - t0) * 1000
        
        try: flow_queue.put({"fx": fx, "fy": fy, "mask": np.zeros(new_gray.shape), "times": t_flow}, timeout=1)
        except: pass
        old_gray = new_gray

def foe_smoke_detector_worker(target_size):
    global latest_smoke_canvas, t_core2, t_core3
    while True:
        try: packet = flow_queue.get(timeout=1)
        except: continue
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        t_foe, t_draw = detect_smoke_motion(packet["fx"], packet["fy"], 10, 0.7, canvas, packet["mask"])
        with data_lock:
            latest_smoke_canvas, t_core2, t_core3 = canvas, packet["times"], t_foe + t_draw
        flow_queue.task_done()

# main() 主程式邏輯維持不變，執行時即可正常運作

# ==========================================
# 【CORE 0 主執行緒】專職：影像讀取、跨核心結果融合、畫面渲染
# ==========================================
def main():
    global current_raw_frame
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "fire_smoke1.mp4")
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
 
    if not os.path.exists(video_path):
        print("找不到影片檔案")
        return

    # 💡 一鍵初始化背景執行緒，分派任務給核心 1, 2, 3
    threading.Thread(target=fire_worker, args=(IMAGE_SIZE,), daemon=True).start()
    if ENABLE_SMOKE_ANALYSIS:
        threading.Thread(target=optical_flow_worker, args=(video_path, IMAGE_SIZE, dilate_kernel), daemon=True).start()
        threading.Thread(target=foe_smoke_detector_worker, args=(IMAGE_SIZE,), daemon=True).start()

    input_video = cv2.VideoCapture(video_path)

    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            input_video.set(cv2.CAP_PROP_POS_FRAMES, 0) # 影片播完自動循環
            continue

        # 核心 0 第一要務：以極快的速度縮放影像，並拷貝分發給其他核心
        resized_frame = cv2.resize(frame, IMAGE_SIZE)
        with data_lock:
            current_raw_frame = resized_frame.copy()

        # 初始化當前畫面的基礎畫布
        display_frame = resized_frame.copy()
        local_t1 = 0.0
        local_t2 = 0.0
        local_t3 = 0.0

        # 非阻塞獲取背景所有核心融合回來的結果圖層
        with data_lock:
            if latest_fire_canvas is not None:
                display_frame = latest_fire_canvas.copy()
                local_t1 = t_fire_branch
            
            if latest_smoke_canvas is not None:
                # 使用 bitwise_or 將核心 3 的藍箭頭、紅點疊加到火焰偵測的成果圖上
                display_frame = cv2.bitwise_or(display_frame, latest_smoke_canvas)
                local_t2 = t_core2
                local_t3= t_core3

        total_cost = local_t1 + local_t2 + local_t3
        cv2.putText(display_frame, f"Core 1:{local_t1} ms", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Core 2:{local_t2} ms", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Core 3:{local_t3} ms", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Total Core: {total_cost} ms", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Output Video", display_frame)

        if cv2.waitKey(10) >= 0:
            break

    input_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()