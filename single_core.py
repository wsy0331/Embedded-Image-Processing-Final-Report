import os
import cv2
import numpy as np
from datetime import datetime
import time
import sys


def detect_fire(
    curr_frame,
    bg_subtractor,
    display_frame,
    kernel=None,
    lower_fire=None,
    upper_fire=None,
):
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if lower_fire is None:
        lower_fire = np.array([0, 100, 150])
    if upper_fire is None:
        upper_fire = np.array([60, 255, 255])

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

    return fire_mask


def detect_smoke(old_gray, new_gray, display_frame, fire_mask, step=10, min_speed=0.7):
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    fx, fy = cv2.split(flow)
    fx = cv2.GaussianBlur(fx.astype(np.float32), (5, 5), 0)
    fy = cv2.GaussianBlur(fy.astype(np.float32), (5, 5), 0)

    height, width = new_gray.shape
    y, x = (
        np.mgrid[step // 2 : height : step, step // 2 : width : step]
        .reshape(2, -1)
        .astype(int)
    )

    fx_sampled, fy_sampled = fx[y, x], fy[y, x]
    magnitude = np.hypot(fx_sampled, fy_sampled)

    valid_idx = (magnitude > min_speed) & (fire_mask[y, x] == 0)

    valid_x, valid_y = x[valid_idx], y[valid_idx]
    valid_fx, valid_fy = fx_sampled[valid_idx], fy_sampled[valid_idx]

    for i in range(len(valid_x)):
        end_x = int(valid_x[i] + valid_fx[i] * 2)
        end_y = int(valid_y[i] + valid_fy[i] * 2)

        if 0 <= end_x < width and 0 <= end_y < height:
            if fire_mask[end_y, end_x] == 0:
                cv2.arrowedLine(
                    display_frame,
                    (valid_x[i], valid_y[i]),
                    (end_x, end_y),
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.5,
                )

    if len(valid_x) >= 2:
        A = np.vstack((valid_fy, -valid_fx)).T
        b = (valid_fy * valid_x - valid_fx * valid_y).reshape(-1, 1)
        try:
            foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            foe_x, foe_y = int(foe[0][0]), int(foe[1][0])
            if 0 <= foe_x < width and 0 <= foe_y < height:
                cv2.circle(display_frame, (foe_x, foe_y), 6, (0, 0, 255), -1)
        except Exception:
            pass


def main():
    # 🌟 已移除：cv2.setNumThreads(1) 與 psutil 核心鎖定指令
    # 讓系統恢復預設狀態：允許 OpenCV 與作業系統自由調用多核心並行加速

    # 接收來自外部總管 B.py 的影片路徑參數
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "fire_smoke11.mp4"
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_videos/output_p_{timestamp}.mp4"  # 調整檔名避免衝突
    target_size = (640, 480)

    input_video = cv2.VideoCapture(video_path)
    if not input_video.isOpened():
        print(f"[ERROR] 無法開啟影片檔案: {video_path}")
        return

    ret, first_frame = input_video.read()
    if not ret:
        print("[ERROR] 無 = 讀取影片第一幀")
        return

    old_resized = cv2.resize(first_frame, target_size)
    old_gray = cv2.cvtColor(old_resized, cv2.COLOR_BGR2GRAY)
    old_gauss = cv2.GaussianBlur(old_gray, (5, 5), 0)

    bg_subtractor_fire = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=8, detectShadows=True
    )
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    writer = None

    try:
        while input_video.isOpened():
            start_time = time.time()  # 記錄單影格開始時間

            ret, frame = input_video.read()
            if not ret:
                break

            new_frame = cv2.resize(frame, target_size)
            raw_display = new_frame.copy()
            new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            new_gauss = cv2.GaussianBlur(new_gray, (5, 5), 0)

            if writer is None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, target_size
                )
                print(f"[INFO] p.py 影片開始錄製，儲存路徑為: {output_path}")

            fire_mask = detect_fire(
                new_frame, bg_subtractor_fire, display_frame=new_frame
            )
            fire_mask_dilated = cv2.dilate(fire_mask, dilate_kernel)
            detect_smoke(
                old_gauss,
                new_gauss,
                display_frame=new_frame,
                fire_mask=fire_mask_dilated,
            )

            end_time = time.time()
            process_time = (end_time - start_time) * 1000  # 轉換為毫秒
            text = f"Process Time: {process_time:.2f} ms"
            cv2.putText(
                new_frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Input Video (Raw)", raw_display)
            cv2.imshow("Output Video (Processed)", new_frame)

            writer.write(new_frame)
            old_gauss = new_gauss.copy()

            if cv2.waitKey(25) >= 0:
                break
    finally:
        input_video.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n--- p.py 視訊處理完畢，已成功儲存至: {output_path} ---")


if __name__ == "__main__":
    main()