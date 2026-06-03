import os
import cv2
import numpy as np

FLOW_THRESHOLD_WINDOW = "Flow Threshold"
FLOW_THRESHOLD_MAX = 200  # Trackbar 的最大值， 200 → 最大門檻 2.0
FLOW_THRESHOLD_INIT = 60  # Trackbar 的初始值， 60 → 最大門檻 0.6
MIN_SMOKE_SPEED = (0.8)  # 光流速度的最小門檻，單位是像素/幀，調大可去除微小雜訊，調小可偵測更緩慢的煙霧
MAX_DOWNWARD_FLOW = 0.05  # 最大向下流動速度，單位是像素/幀，調小可過濾向下的流動，調大寬鬆向下流動的煙霧

def main():
    INPUT_VIDEO = "fire_smoke3.mp4"
    TARGET_SIZE = (640, 480)
    OUTPUT_VIDEO = "output_videos/output_smoke.mp4"

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if cap.isOpened():
        ret, first_frame = cap.read()
        if not ret:
            print("無法讀取第一幀")
            cap.release()
            return
    else:
        print("無法開啟影片檔案")
        return

    # 火焰專用：MOG2 背景相減器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 火焰 HSV 範圍
    lower_fire = np.array([0, 100, 150])
    upper_fire = np.array([35, 255, 255])

    prev_frame = cv2.resize(first_frame, TARGET_SIZE)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # 對用於光流估計的灰階影像先做高斯模糊，並加入中值濾波以去除脈衝雜訊（僅影響煙霧偵測）
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    prev_gray = cv2.medianBlur(prev_gray, 5)
    foe_history = []

    cv2.namedWindow(FLOW_THRESHOLD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "flow_thresh_x100",
        FLOW_THRESHOLD_WINDOW,
        FLOW_THRESHOLD_INIT,
        FLOW_THRESHOLD_MAX,
        lambda value: None,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 檢查並建立輸出資料夾
        os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

        # 延遲初始化 VideoWriter，使用來源影片的 FPS
        if "writer" not in locals():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, TARGET_SIZE)

        curr_frame = cv2.resize(frame, TARGET_SIZE)
        hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # 為了只對煙霧光流使用高斯模糊，對灰階影像做模糊後再做中值濾波，但不影響火焰顏色/遮罩處理
        curr_gray_blur = cv2.GaussianBlur(curr_gray, (5, 5), 1)
        curr_gray_blur = cv2.medianBlur(curr_gray_blur, 5)

        motion_mask = bg_subtractor.apply(curr_frame)
        _, motion_mask = cv2.threshold(motion_mask, 254, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        fire_color_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
        fire_mask = cv2.bitwise_and(motion_mask, fire_color_mask)
        fire_contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray_blur,
            None,
            pyr_scale=0.5,
            levels=1,
            winsize=5,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

        # 估計全域相機位移，並從光流中扣掉（相機補償）
        camera_flow = np.median(flow.reshape(-1, 2), axis=0)
        compensated_flow = flow - camera_flow

        display_frame = curr_frame.copy()

        flow_threshold = (
            cv2.getTrackbarPos("flow_thresh_x100", FLOW_THRESHOLD_WINDOW) / 100.0
        )

        flow_mag = np.hypot(compensated_flow[:, :, 0], compensated_flow[:, :, 1])
        flow_mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
        upward_or_sideways_mask = compensated_flow[:, :, 1] <= MAX_DOWNWARD_FLOW
        flow_mask[(flow_mag > flow_threshold) & upward_or_sideways_mask] = 255

        flow_mask = cv2.morphologyEx(
            flow_mask, cv2.MORPH_CLOSE, smoke_kernel, iterations=2
        )
        flow_mask = cv2.dilate(flow_mask, smoke_kernel, iterations=3)

        smoke_contours, _ = cv2.findContours(
            flow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        main_smoke_contour = None
        if smoke_contours:
            main_smoke_contour = max(smoke_contours, key=cv2.contourArea)

        main_smoke_mask = np.zeros_like(flow_mask)
        if main_smoke_contour is not None and cv2.contourArea(main_smoke_contour) > 20:
            cv2.drawContours(main_smoke_mask, [main_smoke_contour], -1, 255, -1)
        else:
            main_smoke_mask = flow_mask.copy()

        step = 5
        h, w = curr_gray.shape[:2]
        y, x = (
            np.mgrid[step // 2 : h : step, step // 2 : w : step]
            .reshape(2, -1)
            .astype(int)
        )
        fx, fy = compensated_flow[y, x].T

        raw_idx = np.where(main_smoke_mask[y, x] > 0)[0]
        mags = np.hypot(fx[raw_idx], fy[raw_idx])
        candidate_idx = raw_idx[
            (mags > MIN_SMOKE_SPEED) & (fy[raw_idx] <= MAX_DOWNWARD_FLOW)
        ]

        # 估計「煙霧 FOE」
        def compute_foe(idx):
            if idx.size < 3:  # 需要至少三個點來估計交會點
                return None
            try:
                A = np.vstack([fy[idx], -fx[idx]]).T
                b = fy[idx] * x[idx] - fx[idx] * y[idx]
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                return float(sol[0]), float(sol[1])
            except np.linalg.LinAlgError:
                return None

        smoke_foe = compute_foe(candidate_idx)

        averaged_smoke_foe = smoke_foe
        if smoke_foe is not None:
            foe_history.append(smoke_foe)
            if len(foe_history) > 5:
                foe_history.pop(0)

            if len(foe_history) == 5:
                averaged_smoke_foe = (
                    float(np.mean([point[0] for point in foe_history])),
                    float(np.mean([point[1] for point in foe_history])),
                )

        smoke_outline_mask = np.zeros_like(flow_mask)
        if averaged_smoke_foe is not None:
            foe_x, foe_y = averaged_smoke_foe
            for i in candidate_idx:
                startX, startY = x[i], y[i]
                flow_x, flow_y = fx[i], fy[i]
                radial_x = startX - foe_x
                radial_y = startY - foe_y
                radial_norm = np.hypot(radial_x, radial_y)
                flow_norm = np.hypot(flow_x, flow_y)
                if radial_norm > 1 and flow_norm > 0:
                    cosine = (radial_x * flow_x + radial_y * flow_y) / (
                        radial_norm * flow_norm
                    )
                    if cosine > 0.15:
                        cv2.circle(smoke_outline_mask, (startX, startY), 10, 255, -1)

            outline_contours, _ = cv2.findContours(
                smoke_outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if outline_contours:
                smoke_outline = max(outline_contours, key=cv2.contourArea)
                if cv2.contourArea(smoke_outline) > 20:
                    smoke_outline = cv2.convexHull(smoke_outline)
                    cv2.drawContours(display_frame, [smoke_outline], -1, (255, 0, 0), 2)

        smoke_grid_idx = candidate_idx
        flow_scale = 3
        for i in smoke_grid_idx:
            startX, startY = x[i], y[i]
            endX = int(startX + fx[i] * flow_scale)
            endY = int(startY + fy[i] * flow_scale)
            cv2.circle(display_frame, (startX, startY), 1, (255, 0, 0), -1)
            cv2.arrowedLine(
                display_frame,
                (startX, startY),
                (endX, endY),
                (255, 0, 0),
                1,
                tipLength=0.3,
            )

        if averaged_smoke_foe is not None:
            sx, sy = (
                int(round(averaged_smoke_foe[0])),
                int(round(averaged_smoke_foe[1])),
            )
            if 0 < sx < 640 and 0 < sy < 480:
                cv2.circle(display_frame, (sx, sy), 7, (0, 0, 255), -1)
                cv2.putText(
                    display_frame,
                    "FoE",
                    (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

        for contour in fire_contours:
            if cv2.contourArea(contour) > 15:
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)

        # 將處理後畫面寫入輸出影片
        if "writer" in locals() and writer is not None:
            writer.write(display_frame)

        cv2.imshow("Fire (MOG2) & Smoke (Optical Flow)", display_frame)
        # 將模糊（高斯 + 中值）後的灰階影像作為下一幀的 prev_gray（只影響光流）
        prev_gray = curr_gray_blur.copy()

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    if "writer" in locals() and writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
