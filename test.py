import cv2
import numpy as np
import os

cap = cv2.VideoCapture("fire_smoke1.mp4")
target_w, target_h = 1280, 720

# 取得原始視頻的 FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# 建立輸出文件夾
output_folder = "output_videos"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化視頻寫入器
output_path = os.path.join(output_folder, "fire_detection_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

# 初始化
ret, frame = cap.read()
prev_frame = cv2.resize(frame, (target_w, target_h))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_frame = cv2.resize(frame, (target_w, target_h))
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # --- 步驟 1: 光流檢測運動區域 ---
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=7,
        winsize=5,
        iterations=10,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )

    # 計算光流的幅度（運動強度）
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 創建運動遮罩（只保留有明顯運動的區域）
    motion_threshold = 2.0  # 運動強度閾值
    motion_mask = cv2.inRange(mag, motion_threshold, 255)

    # 形態學操作清潔運動遮罩
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    # --- 步驟 2: 使用 RGB 通道比例檢測紅色/橙色火焰 ---
    # 分解 BGR 通道（OpenCV 是 BGR 順序）
    b = curr_frame[:, :, 0].astype(np.float32)
    g = curr_frame[:, :, 1].astype(np.float32)
    r = curr_frame[:, :, 2].astype(np.float32)

    # 避免除以零
    g_safe = np.where(g > 10, g, 1)

    # 計算紅/綠比例（火焰特徵：紅色通道遠高於綠色）
    rg_ratio = r / g_safe

    # 計算紅/藍比例（火焰特徵：紅色通道高於藍色）
    b_safe = np.where(b > 1, b, 1)
    rb_ratio = r / b_safe

    # 火焰判定：R/G > 1.4 且 R/B > 1.2 且 R 值足夠高
    fire_color_mask = (rg_ratio > 1) & (rb_ratio > 1.2) & (r > 120)
    fire_color_mask = fire_color_mask.astype(np.uint8) * 255

    # 形態學操作清潔火焰顏色遮罩
    fire_color_mask = cv2.morphologyEx(fire_color_mask, cv2.MORPH_CLOSE, kernel)
    fire_color_mask = cv2.morphologyEx(fire_color_mask, cv2.MORPH_OPEN, kernel)

    # --- 步驟 3: 結合運動 + 顏色 ---
    # 只保留既有運動又是火焰顏色的區域
    combined_mask = cv2.bitwise_and(motion_mask, fire_color_mask)

    # --- 步驟 4:標記火源 ---
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- 步驟 5: 在原始視頻上繪製光流向量 ---
    display_frame = curr_frame.copy()
    step = 10  # 光流採樣步長（越小越密集）
    h, w = curr_frame.shape[:2]
    y_coords, x_coords = (
        np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    )
    fx, fy = flow[y_coords, x_coords].T

    # 繪製光流箭頭（綠色）
    for i in range(len(x_coords)):
        if np.sqrt(fx[i] ** 2 + fy[i] ** 2) > 0.5:  # 流動閾值（越小越多箭頭）
            x, y = x_coords[i], y_coords[i]
            cv2.arrowedLine(
                display_frame,
                (x, y),
                (int(x + fx[i] * 2), int(y + fy[i] * 2)),
                (0, 255, 0),
                1,
                tipLength=0.3,
            )

    fire_sources = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # 只處理足夠大的區域
        if area > 10:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                fire_sources.append((cx, cy, area))

                # 用紅色加粗描邊標記火焰輪廓
                cv2.drawContours(display_frame, [contour], 0, (0, 0, 255), 3)
                # 標記面積
                cv2.putText(
                    display_frame,
                    f"Area:{int(area)}",
                    (cx - 30, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # --- 顯示統計信息 ---
    cv2.putText(
        display_frame,
        f"Fire Sources (Motion + Color): {len(fire_sources)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    # 顯示運動強度範圍
    cv2.putText(
        display_frame,
        f"Motion: {motion_mask.sum() // 255} pixels",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    cv2.imshow("Fire Detection - Motion + Color + Optical Flow", display_frame)
    cv2.imshow("Motion Mask (Optical Flow)", motion_mask)
    cv2.imshow("Fire Color Mask (R/G > 1.4 & R/B > 1.2)", fire_color_mask)
    cv2.imshow("Combined (Motion + Color)", combined_mask)

    # 寫入處理後的幀到輸出視頻
    out.write(display_frame)

    prev_gray = curr_gray
    prev_frame = curr_frame

    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"視頻已保存到: {output_path}")
