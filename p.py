import os
import cv2
import numpy as np

def detect_fire(curr_frame, bg_subtractor, kernel, lower_fire, upper_fire, display_frame):
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

def detect_smoke(old_gray, new_gray, step, min_speed, display_frame, fire_mask):
    # 1. 計算光流
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    fx, fy = cv2.split(flow)
    # 高斯濾波對 fx 和 fy 進行平滑，sigma 為 0 時會根據 kernel 大小自動計算
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

    # 2. 過濾火源區域與過低速度
    valid_idx = (magnitude > min_speed) & (fire_mask[y, x] == 0)

    # 3. 過濾剛性移動 (向量散度過低)
    if (
        np.var(fx_sampled[valid_idx]) < 0.5
        and np.var(fy_sampled[valid_idx]) < 0.5
    ):
        return

    valid_x, valid_y = x[valid_idx], y[valid_idx]
    valid_fx, valid_fy = fx_sampled[valid_idx], fy_sampled[valid_idx]

    # 4. 繪製箭頭並進行邊界檢查
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

    # 5. 計算並繪製 FoE
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
    video_path = "fire_smoke5.mp4"
    output_path = "output_videos/output_combined.mp4"
    input_video = cv2.VideoCapture(video_path)

    if not input_video.isOpened():
        return

    ret, first_frame = input_video.read()
    target_size = (640, 480)
    old_gray = cv2.GaussianBlur(
        cv2.cvtColor(cv2.resize(first_frame, target_size), cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )
    bg_subtractor_fire = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    writer = None

    while input_video.isOpened():
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
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                target_size,
            )

        fire_mask = detect_fire(
            curr_frame=new_frame,
            bg_subtractor=bg_subtractor_fire,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            lower_fire=np.array([0, 100, 150]),
            upper_fire=np.array([60, 255, 255]),
            display_frame=new_frame,
        )

        fire_mask_dilated = cv2.dilate(fire_mask, dilate_kernel)

        detect_smoke(old_gray, new_gauss, 10, 0.7, new_frame, fire_mask_dilated)

        cv2.imshow("Input Video", raw_display)
        cv2.imshow("Output Video", new_frame)
        writer.write(new_frame)
        old_gray = new_gauss.copy()

        if cv2.waitKey(25) >= 0:
            break

    input_video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()