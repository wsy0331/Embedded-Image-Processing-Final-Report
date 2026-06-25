import os, cv2, time, threading, queue  # noqa: E401
import numpy as np
from datetime import datetime  # 🌟 引入時間戳記模組
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
        with data_lock: 
            frame = current_raw_frame.copy() if current_raw_frame is not None else None
            
        if frame is None: 
            time.sleep(0.001)
            continue
            
        canvas = frame.copy()
        t0 = time.perf_counter()
        detect_fire(frame, bg, k, np.array([0,100,150]), np.array([60,255,255]), canvas)
        with data_lock: 
            latest_fire_canvas, t_fire_branch = canvas, (time.perf_counter()-t0)*1000

def optical_flow_worker(video_path, target_size, dilate_kernel):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret: return
    old_gray = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2GRAY)
    
    height, width = target_size[1], target_size[0]
    mid_y = height // 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        new_gray = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2GRAY)
        
        t0 = time.perf_counter()
        results = [None, None]
        
        def compute_top_half():
            results[0] = cv2.calcOpticalFlowFarneback(
                old_gray[0:mid_y, :], new_gray[0:mid_y, :], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
        def compute_bottom_half():
            results[1] = cv2.calcOpticalFlowFarneback(
                old_gray[mid_y:height, :], new_gray[mid_y:height, :], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        thread_top = threading.Thread(target=compute_top_half)
        thread_bottom = threading.Thread(target=compute_bottom_half)
        
        thread_top.start()
        thread_bottom.start()
        
        thread_top.join()
        thread_bottom.join()
        
        flow = np.vstack((results[0], results[1]))
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

# ==========================================
# 【CORE 0 主執行緒】專職：影像讀取、跨核心結果融合、畫面渲染與影片錄製
# ==========================================
def main():
    global current_raw_frame
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(current_dir, "fire_smoke11.mp4")
        
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
 
    if not os.path.exists(video_path):
        print(f"找不到影片檔案: {video_path}")
        return

    # 🌟 初始化影片錄製路徑設定 (專屬 A.py 的檔名)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_videos/output_A_{timestamp}.mp4"
    writer = None

    threading.Thread(target=fire_worker, args=(IMAGE_SIZE,), daemon=True).start()
    if ENABLE_SMOKE_ANALYSIS:
        threading.Thread(target=optical_flow_worker, args=(video_path, IMAGE_SIZE, dilate_kernel), daemon=True).start()
        threading.Thread(target=foe_smoke_detector_worker, args=(IMAGE_SIZE,), daemon=True).start()

    input_video = cv2.VideoCapture(video_path)

    try:  # 🌟 加上 try...finally 機制
        while input_video.isOpened():
            t0_core0 = time.perf_counter()

            ret, frame = input_video.read()
            if not ret:
                input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            resized_frame = cv2.resize(frame, IMAGE_SIZE)
            with data_lock:
                current_raw_frame = resized_frame.copy()

            display_frame = resized_frame.copy()
            local_t1 = 0.0
            local_t2 = 0.0
            local_t3 = 0.0

            with data_lock:
                if latest_fire_canvas is not None:
                    display_frame = latest_fire_canvas.copy()
                    local_t1 = t_fire_branch
                
                if latest_smoke_canvas is not None:
                    smoke_max = np.max(latest_smoke_canvas, axis=2)
                    alpha = smoke_max.astype(np.float32) / 255.0
                    alpha = cv2.merge([alpha, alpha, alpha])

                    foreground = latest_smoke_canvas.astype(np.float32)
                    background = display_frame.astype(np.float32)
                    
                    display_frame = (foreground * alpha + background * (1.0 - alpha)).astype(np.uint8)
                    
                    local_t2 = t_core2
                    local_t3 = t_core3

            local_t0 = (time.perf_counter() - t0_core0) * 1000
            process_time = local_t0 + local_t1 + local_t2 + local_t3

            # 繪製面板
            cv2.putText(display_frame, f"Core 0 (Main/Render): {local_t0:.2f} ms", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, f"Core 1 (Fire Sync):   {local_t1:.2f} ms", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, f"Core 2 (Flow 2-Core): {local_t2:.2f} ms", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, f"Core 3 (FoE Async):   {local_t3:.2f} ms", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, f"Process Time: {process_time:.2f} ms", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 🌟 延遲初始化 VideoWriter
            if writer is None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, IMAGE_SIZE
                )
                print(f"[INFO] A.py 影片開始錄製，儲存路徑為: {output_path}")

            cv2.imshow("Output Video", display_frame)
            
            # 🌟 寫入當前整合完畢的影像
            writer.write(display_frame)

            if cv2.waitKey(10) >= 0:
                break
    finally:
        # 🌟 資源安全釋放
        input_video.release()
        if writer:
            writer.release()
            print(f"[INFO] A.py 視訊儲存完畢: {output_path}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()