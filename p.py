import cv2
import numpy as np
import os


def detect_fire(
    curr_frame, bg_subtractor, kernel, lower_fire, upper_fire, display_frame
):
    """
    【火焰辨識模組】
    透過背景相減與 HSV 顏色特徵交集，偵測火焰並繪製綠色輪廓。
    """
    motion_mask = bg_subtractor.apply(curr_frame)
    _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    fire_color_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)

    fire_mask = cv2.bitwise_and(motion_mask, fire_color_mask)

    fire_contours, _ = cv2.findContours(
        fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in fire_contours:
        if cv2.contourArea(contour) > 15:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)

    return fire_contours


def detect_smoke(old_gray, new_gray, step, min_speed, display_frame):
    """
    【煙霧光流辨識與繪製模組】

    參數:
    - old_gray: 前一幀的灰階影像 (已做高斯模糊)
    - new_gray: 目前幀的灰階影像 (已做高斯模糊)
    - step: 稀疏網格抽樣的間距 (像素)
    - min_speed: 光流速度的最小門檻 (即原本的 magnitude > 0.5)
    - display_frame: 要在上面畫藍色光流箭頭的輸出影像幀

    回傳:
    - flow: 計算出來的原始稠密光流數據 (供後續進階分析使用)
    """
    # 1. 計算 Farneback 稠密光流
    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        new_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    height, width = new_gray.shape

    # 2. 高效網格抽樣 (NumPy 矩陣向量化操作)
    y, x = (
        np.mgrid[step // 2 : height : step, step // 2 : width : step]
        .reshape(2, -1)
        .astype(int)
    )
    fx, fy = flow[y, x].T

    # 3. 計算每個抽樣點的運動速度 (向量長度)
    magnitude = np.hypot(fx, fy)

    # 4. 篩選出運動速度大於門檻的點
    valid_idx = magnitude > min_speed

    # 5. 疊代篩選後的點，繪製藍色光流箭頭
    for i in np.where(valid_idx)[0]:
        start_point = (x[i], y[i])
        end_point = (int(x[i] + fx[i] * 2), int(y[i] + fy[i] * 2))
        # 繪製藍色動態箭頭
        cv2.arrowedLine(
            display_frame,
            start_point,
            end_point,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
            tipLength=0.5,
        )
    return flow


def main():
    video_path = "fire_smoke3.mp4"
    INPUT_VIDEO = cv2.VideoCapture(video_path)
    OUTPUT_VIDEO = "output_videos/output_smoke.mp4"

    if not INPUT_VIDEO.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    ret, first_frame = INPUT_VIDEO.read()
    if not ret:
        print("無法讀取影片第一幀")
        INPUT_VIDEO.release()
        return

    # 統一解析度與初始化第一幀
    TARGET_SIZE = (640, 480)
    old_frame = cv2.resize(first_frame, TARGET_SIZE)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

    # 2. 初始化火焰偵測參數
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lower_fire = np.array([0, 100, 150])
    upper_fire = np.array([35, 255, 255])

    # 3. 初始化煙霧光流參數
    STEP = 10  # 網格抽樣間距
    MIN_SPEED = 0.5  # 抑制噪點的最小速度門檻

    while INPUT_VIDEO.isOpened():
        ret, frame = INPUT_VIDEO.read()
        if not ret:
            print("影片讀取完畢或發生錯誤")
            break
        # 檢查並建立輸出資料夾
        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

        # 延遲初始化 VideoWriter，使用來源影片的 FPS
        if "writer" not in locals():
            fps = INPUT_VIDEO.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, TARGET_SIZE)

        # 當前影格前置處理
        new_frame = cv2.resize(frame, TARGET_SIZE)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.GaussianBlur(new_gray, (5, 5), 0)

        # 呼叫火焰辨識功能
        detect_fire(new_frame, bg_subtractor, kernel, lower_fire, upper_fire, new_frame)

        # 呼叫煙霧光流功能
        detect_smoke(old_gray, new_gray, STEP, MIN_SPEED, new_frame)

        # 顯示最終結果
        cv2.imshow("Output_VIDEO", new_frame)

        # 將處理後的影像寫入輸出影片
        writer.write(new_frame)

        # 更新前一幀
        old_gray = new_gray.copy()

        if cv2.waitKey(25) >= 0:
            break

    INPUT_VIDEO.release()
    if "writer" in locals() and writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
