import cv2
import numpy as np

def main():
    INPUT_VIDEO = "fire_smoke3.mp4"
    TARGET_SIZE = (640, 480)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    # 火焰專用：MOG2 背景相減器 (運算快，適合抓高亮度火焰)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )
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
        fire_contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ==========================================
        # 模組 2：煙霧偵測 (稠密光流 Farneback + HSV)
        # ==========================================
        # 1. 計算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=1,
            winsize=5,
            iterations=3,
            poly_n=7,
            poly_sigma=1.2,
            flags=0,
        )

        # 繪製結果
        display_frame = curr_frame.copy()

        # 畫火焰
        for contour in fire_contours:
            if cv2.contourArea(contour) > 15:
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)

        # 畫煙霧
        step = 10  # 網格間距
        scale = 3.0 # 箭頭更明顯
        h, w = prev_gray.shape[:2]
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + (fx * scale), y + (fy * scale)]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        valid_lines = []
        for i, (startX, startY) in enumerate(zip(x, y)):
            if abs(fx[i]) > 1 or abs(fy[i]) > 1:
                valid_lines.append(lines[i])
        for line in valid_lines:
            pt1 = (int(line[0][0]), int(line[0][1])) 
            pt2 = (int(line[1][0]), int(line[1][1])) 
            cv2.arrowedLine(display_frame, pt1, pt2, (255, 0, 0), 1, tipLength=0.5) # tipLength控制「三角箭頭」大小

        cv2.imshow("Fire (MOG2) & Smoke (Optical Flow)", display_frame)
        # 更新上一幀影像，供下一次光流比對使用
        prev_gray = curr_gray.copy()

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()