import cv2
import numpy as np


def main():
    INPUT_VIDEO = "fire_smoke1.mp4"
    TARGET_SIZE = (640, 480)
    FLOW_THRESHOLD_WINDOW = "Flow Threshold"
    FLOW_THRESHOLD_MAX = 200
    FLOW_THRESHOLD_INIT = 45

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
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

    ret, first_frame = cap.read()
    if not ret:
        print("無法讀取第一幀")
        cap.release()
        return

    prev_frame = cv2.resize(first_frame, TARGET_SIZE)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
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

        curr_frame = cv2.resize(frame, TARGET_SIZE)
        hsv_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

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
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=1,
            winsize=5,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

        display_frame = curr_frame.copy()

        flow_threshold = (
            cv2.getTrackbarPos("flow_thresh_x100", FLOW_THRESHOLD_WINDOW) / 100.0
        )

        flow_mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
        flow_mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
        flow_mask[flow_mag > flow_threshold] = 255

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
        fx, fy = flow[y, x].T

        raw_idx = np.where(main_smoke_mask[y, x] > 0)[0]
        mags = np.hypot(fx[raw_idx], fy[raw_idx])
        candidate_idx = raw_idx[mags > 0.1]

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

        cv2.imshow("Fire (MOG2) & Smoke (Optical Flow)", display_frame)
        prev_gray = curr_gray.copy()

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
