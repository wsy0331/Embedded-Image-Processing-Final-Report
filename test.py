import cv2
import numpy as np

def main():
    INPUT_VIDEO = "fire_smoke3.mp4"
    TARGET_SIZE = (640, 480)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    # 火焰專用：MOG2 背景相減器 (運算快，適合抓高亮度火焰)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ==========================================
    # 初始化：讀取第一幀給光流法使用
    # ==========================================
    ret, first_frame = cap.read()
    if not ret:
        print("無法讀取影片")
        return
    prev_frame = cv2.resize(first_frame, TARGET_SIZE)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # 定義 HSV 顏色範圍
    # ==========================================
    # 火焰 (紅橘、高飽和、高亮度)
    lower_fire = np.array([0, 100, 150])
    upper_fire = np.array([35, 255, 255])

    # 煙霧 (深灰到純白)
    lower_smoke = np.array([0, 0, 20])   
    upper_smoke = np.array([179, 45, 255]) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame = cv2.resize(frame, TARGET_SIZE) 
        hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # 模組 1：火焰偵測 (MOG2 + HSV)
        # ==========================================
        motion_mask = bg_subtractor.apply(curr_frame)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        fire_color_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
        fire_mask = cv2.bitwise_and(motion_mask, fire_color_mask)
        fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ==========================================
        # 模組 2：煙霧偵測 (稠密光流 Farneback + HSV)
        # ==========================================
        # 1. 計算稠密光流
        # 為了降低樹莓派負擔，這裡的層數(levels)設為 3，迭代次數(iterations)設為 3
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=5, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # 2. 計算光流的移動強度 (Magnitude)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 3. 建立光流運動遮罩 (過濾掉極微小的相機雜訊，保留移動強度 > 1.0 的像素)
        flow_motion_mask = cv2.inRange(mag, 1.0, 255)
        flow_motion_mask = cv2.morphologyEx(flow_motion_mask, cv2.MORPH_OPEN, kernel)

        # 4. 取得煙霧顏色遮罩
        smoke_color_mask = cv2.inRange(hsv_frame, lower_smoke, upper_smoke)

        # 5. 結合：光流偵測到的動態 + 煙霧的 HSV 顏色
        smoke_mask = cv2.bitwise_and(flow_motion_mask, smoke_color_mask)
        
        # 確保煙霧不會標記到火焰的區域
        final_smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(fire_color_mask))

        # 繪製結果
        display_frame = curr_frame.copy()

        # 畫火焰 (綠色框線)
        for contour in fire_contours:
            if cv2.contourArea(contour) > 15:
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)

        # 畫煙霧 (藍色框線)
        smoke_contours, _ = cv2.findContours(final_smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in smoke_contours:
            if cv2.contourArea(contour) > 200: # 煙霧面積通常較大
                cv2.drawContours(display_frame, [contour], -1, (255, 0, 0), 2)

        cv2.imshow("Fire (MOG2) & Smoke (Optical Flow)", display_frame)
        cv2.imshow("Optical Flow Mask", flow_motion_mask) # 讓你看光流抓動態的效果

        # 更新上一幀影像，供下一次光流比對使用
        prev_gray = curr_gray.copy()

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()