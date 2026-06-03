import cv2
import numpy as np
import os

def detect_fire(curr_frame, bg_subtractor, kernel, lower_fire, upper_fire, display_frame):
    """
    【模組 1：火焰辨識】
    透過背景相減與 HSV 顏色特徵交集，偵測火焰並繪製綠色輪廓。
    """
    motion_mask = bg_subtractor.apply(curr_frame)
    _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    fire_color_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
    fire_mask = cv2.bitwise_and(motion_mask, fire_color_mask)

    fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in fire_contours:
        if cv2.contourArea(contour) > 15:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)

    return fire_contours


def detect_smoke(old_gray, new_gray, step, min_speed, display_frame):
    """
    【模組 2：煙霧光流與 FoE 辨識】 (保留您原本的功能)
    繪製藍色箭頭與紅色 FoE 起火點。
    """
    # 1. 計算 Farneback 稠密光流
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, new_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    height, width = new_gray.shape

    # 2. 高效網格抽樣
    y, x = (
        np.mgrid[step // 2 : height : step, step // 2 : width : step]
        .reshape(2, -1)
        .astype(int)
    )
    fx, fy = flow[y, x].T

    # 3. 計算每個抽樣點的運動速度
    magnitude = np.hypot(fx, fy)

    # 4. 篩選出運動速度大於門檻的點
    valid_idx = magnitude > min_speed
    valid_x = x[valid_idx]
    valid_y = y[valid_idx]
    valid_fx = fx[valid_idx]
    valid_fy = fy[valid_idx]

    # 5. 繪製藍色光流箭頭
    for i in range(len(valid_x)):
        start_point = (valid_x[i], valid_y[i])
        end_point = (int(valid_x[i] + valid_fx[i] * 2), int(valid_y[i] + valid_fy[i] * 2))
        cv2.arrowedLine(
            display_frame, start_point, end_point,
            (255, 0, 0), 1, cv2.LINE_AA, tipLength=0.5,
        )

    # 6. 計算並繪製 FoE (擴張焦點)
    if len(valid_x) >= 2:
        A = np.vstack((valid_fy, -valid_fx)).T
        b = (valid_fy * valid_x - valid_fx * valid_y).reshape(-1, 1)
        try:
            foe, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            foe_x, foe_y = int(foe[0][0]), int(foe[1][0])
            # 防呆：避免 FoE 計算到畫面外
            if -1000 < foe_x < 680 + 1000 and -1000 < foe_y < 480 + 1000:
                # 畫出 FoE 的實心紅點
                cv2.circle(display_frame, (foe_x, foe_y), 6, (0, 0, 255), -1)
        except Exception:
            pass

    return flow


def detect_smoke_area(curr_frame, bg_subtractor_smoke, kernel_small, display_frame):
    # 1. 取得前景遮罩
    smoke_mask = bg_subtractor_smoke.apply(curr_frame)
    _, smoke_mask = cv2.threshold(smoke_mask, 254, 255, cv2.THRESH_BINARY)
    # 2. 形態學處理：用極大 Kernel 縫合破碎的煙霧斑塊
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel_small)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel_small)
  

    # 3. 尋找輪廓與繪製凸包
    smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in smoke_contours:
        if cv2.contourArea(contour) >3: 
            cv2.drawContours(display_frame, [contour], -1, (0, 165, 255), 2)

def main():
    INPUT_VIDEO = cv2.VideoCapture("fire_smoke1.mp4")
    OUTPUT_VIDEO = "output_videos/output_combined.mp4"

    if not INPUT_VIDEO.isOpened():
        print("無法開啟影片")
        return

    ret, first_frame = INPUT_VIDEO.read()
    if not ret:
        print("無法讀取影片第一幀")
        INPUT_VIDEO.release()
        return

    # 1. 統一解析度與初始化第一幀
    TARGET_SIZE = (640, 480)
    old_frame = cv2.resize(first_frame, TARGET_SIZE)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

    # 2. 初始化各式偵測參數
    # 火焰專用 (MOG2 較嚴格，需要配合 HSV)
    bg_subtractor_fire = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    kernel_fire = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lower_fire = np.array([0, 100, 150])
    upper_fire = np.array([35, 255, 255])

    # 煙霧範圍專用 (MOG2 較敏感，因為煙霧顏色淡)
    bg_subtractor_smoke = cv2.createBackgroundSubtractorMOG2(history=900, varThreshold=8, detectShadows=True)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # -- 光流專用參數
    STEP = 10
    MIN_SPEED = 0.5

    # 3. 進入影像處理迴圈
    while INPUT_VIDEO.isOpened():
        ret, frame = INPUT_VIDEO.read()
        if not ret:
            print("影片讀取完畢或發生錯誤")
            break
            
        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

        if "writer" not in locals():
            fps = INPUT_VIDEO.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, TARGET_SIZE)

        # 預處理當前幀
        new_frame = cv2.resize(frame, TARGET_SIZE)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.GaussianBlur(new_gray, (5, 5), 0)

        # 1. 畫出火焰綠框
        detect_fire(new_frame, bg_subtractor_fire, kernel_fire, lower_fire, upper_fire, new_frame)
        # 2. 畫出煙霧橘色範圍框
        detect_smoke_area(new_frame, bg_subtractor_smoke, kernel_small, new_frame)
        # 3. 畫出煙霧藍色光流箭頭與紅色 FoE 起火點
        detect_smoke(old_gray, new_gray, STEP, MIN_SPEED, new_frame)

        # 顯示最終畫面
        cv2.imshow("Combined Video Analytics", new_frame)

        # 寫入影片
        writer.write(new_frame)

        # 更新前一幀
        old_gray = new_gray.copy()

        if cv2.waitKey(25) >= 0:
            break

    # 4. 資源釋放
    INPUT_VIDEO.release()
    if "writer" in locals() and writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()