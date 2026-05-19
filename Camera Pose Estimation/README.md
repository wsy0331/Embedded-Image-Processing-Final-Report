
---

# Embedded-Image-Processing-Final-Report

## 主題：室內手機相機姿態估測 Camera Pose Estimation (更新版)

本專案旨在針對單張室內圖像，利用電腦視覺技術估算相機在拍攝當下的世界座標姿態資訊：**Yaw（偏航）、Pitch（俯仰）、Roll（翻滾）**。
本版本強化了線段合併（Line Merging）與消失點（Vanishing Point）聚類邏輯，並支援自動擴展畫布以視覺化位於圖像外的消失點。

---

## 1. 需求 Requirement

### 1.1 功能需求
- **多相容讀取**：支援中文路徑與不同通道格式（RGB/RGBA/Gray）。
- **自適應前處理**：包含 Resize、高斯模糊及基於中位數的自適應 Canny 邊緣偵測。
- **進階線段處理**：
    - 使用機率 Hough 轉換偵測直線。
    - **線段合併系統**：計算線段間的夾角與最短距離，合併重疊或鄰近的雜訊線。
- **消失點聚類**：計算所有線段交點，並透過聚類（Clustering）過濾雜訊，找出信心度最高的消失點（VP）。
- **姿態估算**：包含 Roll、Yaw、Pitch 的自動反算。
- **動態視覺化**：支援畫布自動擴張，完整呈現影像外的消失點與延伸輔助線。

### 1.2 效能需求
- **運算速度**：單張 640x480 圖片處理時間應小於 1 秒。
- **準確度目標**：Roll 誤差 < 5 度；Yaw/Pitch 誤差 < 10 度。

---

## 2. 分析 Analysis

### 2.1 大問題拆解 (Problem Decomposition)
姿態估測任務可進一步拆解為以下關鍵子問題，並由對應的模組解決：

1.  **特徵碎片化問題**：Hough 轉換在複雜場景中會產生大量斷裂、重疊的線段。
    *   *對策*：引入 **`merge_similar_lines`**，利用幾何距離與夾角閾值進行線段融合。
2.  **消失點 (VP) 的穩定性**：任兩條不平行線皆有交點，導致雜訊交點遠多於真實消失點。
    *   *對策*：開發 **Support-based Clustering**，找出擁有最多線段支持的交點群聚區域。
3.  **姿態參數解算**：如何從二維特徵映射回三維旋轉角。
    *   *對策*：利用曼哈頓世界假設（Manhattan World），將垂直線與水平線分別對應至相機的主軸，並透過 $f$ (焦距) 進行反算。
4.  **視覺化範圍限制**：消失點通常落在影像邊框外，傳統顯示法無法觀察其幾何收斂性。
    *   *對策*：設計 **自動擴張畫布演算法**，計算 VP 與原圖的偏移量並動態調整輸出場景。

### 2.2 主要假設 (Key Assumptions)
*   **曼哈頓世界假設**：假設場景中存在三組互相垂直的平行線（地平線、牆角線、天花板邊緣）。
*   **相機內參近似**：假設相機主點（Principal Point）位於影像中心，且焦距 $f \approx 0.8 \times \text{Width}$。
*   **透視投影模型**：影像符合針孔相機模型，無嚴重的徑向畸變（或是畸變已在外部校正）。

### 2.3 風險與挑戰 (Risks & Challenges)
| 風險類型 | 具體描述 | 程式應對方案 |
| :--- | :--- | :--- |
| **雜訊干擾** | 地毯紋路、陰影產生的斜線導致 VP 偏移。 | 使用**中位數統計 (Median)** 取代平均值，強化抗干擾能力。 |
| **幾何退化** | 當拍攝角度極端（如俯拍地面）時，平行線特徵會消失。 | 系統回報預設值，並可透過線段支持度判斷信心分數。 |
| **效能瓶頸** | 計算所有線段兩兩交點的時間複雜度為 $O(N^2)$。 | 先進行**線段合併**大幅降低 $N$ 值，確保單張處理 < 1s。 |
| **顯示誤差** | 手機焦距與預設值不符導致角度偏差。 | 提供 `fx, fy` 參數介面供使用者針對不同機型手動校正。 |

### 2.4 技術指標分析 (Metrics Analysis)
*   **穩定性 (Robustness)**：透過 `distance_segment_to_segment` 計算兩線段間的最短距離，比單純判斷中點距離更能精確捕捉「共線但斷裂」的結構特徵。
*   **視覺化精度**：`get_line_boundary_points` 確保了延伸線（Extended Lines）能精準地從原圖邊界延伸至無窮遠處的消失點，這對於驗證演算法的幾何正確性至關重要。
根據您提供的筆記圖片（![image1](image1)），我為您整理了 **2.5 可使用的工具** 章節。

這部分採用勾選格式，明確標示出本專案在實作中具體採用了哪些技術，以及哪些是作為備選或分析工具：

---


### 2.5 可使用的工具 (修正版)

- [x] **Gaussian Blur (高斯模糊)**：用於前處理平滑影像，濾除高頻雜訊。
- [ ] **Binarization (histogram) (二值化)**：未直接使用，由 Canny 內部的雙閾值機制取代。
- [ ] **Sobel (索貝爾算子)**：未直接使用，改用偵測結果更細緻的 Canny。
- [x] **Canny (坎尼邊緣偵測)**：用於提取場景結構的邊緣特徵。
- [ ] **Contour (輪廓偵測)**：未採用，因室內結構分析主要依賴「線段」而非「封閉輪廓」。
- [x] **Hough (霍夫轉換)**：核心工具，用於將邊緣點集合轉換為具備幾何意義的直線方程式。
- [ ] **Perspective Transform (透視變換)**：
    *   *說明*：**未直接執行影像扭曲（Warping）**。但在姿態解算邏輯中，運用了其**逆透視投影（Inverse Perspective Projection）理論**，將二維消失點座標映射回三維空間的角度。
- [x] **Reference pt. (參考點)**：
    *   *用途*：以 **消失點 (Vanishing Point)** 與 **影像中心點 (Principal Point)** 作為解算相機姿態的幾何參考基準。



---

## 3. 設計 Design
---

### 3.1 系統流程圖 (System Flowchart)

#### **A. 大綱版 (High-Level Overview)**
快速理解系統從輸入到輸出的四個主要階段。

```mermaid
graph LR
    A[影像輸入] --> B[特徵提取與前處理]
    B --> C[幾何姿態解算]
    C --> D[可視化與結果儲存]
```

---

#### **B. 詳細版 (Detailed Pipeline)**
展示程式碼中各個關鍵函式（如線段合併、VP 聚類、畫布擴張）的執行順序與邏輯。

```mermaid
graph TD
    subgraph "1. 影像前處理 (Preprocessing)"
        A[讀取圖片 read_image] --> B[調整尺寸 Resize 640x480]
        B --> C[灰階化 & 高斯模糊]
        C --> D[自適應 Canny 邊緣偵測]
    end

    subgraph "2. 直線偵測與優化 (Line Logic)"
        D --> E[HoughLinesP 直線偵測]
        E --> F[計算線段長度與角度]
        F --> G{線段合併模組}
        G -- "夾角 < 10° & 距離 < 40px" --> H[merge_similar_lines]
        H --> I[取得精簡特徵線段]
    end

    subgraph "3. 幾何姿態解算 (Pose Solver)"
        I --> J[計算所有線段交點 intersection]
        J --> K[消失點聚類 find_vanishing_point]
        K -- "計算最大支持數聚類" --> L[提取最優 VP 座標]
        L --> M[計算 Yaw & Pitch: 基於 VP 位移]
        I --> N[計算 Roll: 水平線中位數角度]
    end

    subgraph "4. 視覺化引擎 (Output Engine)"
        M & N --> O[判斷 VP 是否在圖外]
        O --> P[自動擴展畫布尺寸 draw_result]
        P --> Q[繪製延伸輔助線 & 相機三軸]
        Q --> R[疊加姿態數據文字]
        R --> S[儲存結果圖並顯示]
    end
```

---


### 3.2 模組設計 (新增幾何運算模組)

| 模組分類 | 函式名稱 | 功能描述 |
| :--- | :--- | :--- |
| **基礎 IO** | `read_image` | 支援中文路徑讀取，統一格式為 BGR。 |
| | `cv2_imshow` | (來自 Colab 補丁) 解決雲端環境顯示限制。 |
| **影像處理** | `preprocess` | 進行縮放、模糊化與自適應 Canny 邊緣偵測。 |
| | `detect_hough_lines` | 以機率 Hough 轉換偵測直線，計算長度與初始角度。 |
| **幾何運算** | `dist_points` | 計算空間中兩點間的歐幾里德距離。 |
| | `distance_point_to_segment` | 計算「點」到「線段」的最短距離（垂直距離）。 |
| | `distance_segment_to_segment` | **關鍵模組**：計算兩線段間的最短距離。 |
| | `merge_similar_lines` | **核心邏輯**：合併相似角度且距離鄰近的線段。 |
| | `intersection` | 計算兩直線之交點座標。 |
| | `find_vanishing_point` | 運用聚類演算法與中位數，在大量交點中提取 VP。 |
| | `get_line_boundary_points` | 計算直線與畫布邊界的交點，用於繪製延伸線。 |
| **姿態解算** | `estimate_roll` | 利用水平線段群組估計 Roll 角。 |
| | `estimate_yaw_pitch` | 根據 VP 座標與預設相機內參計算 Yaw 與 Pitch。 |
| **視覺化** | `draw_camera_axis` | 繪製 3D 相機座標軸 (X, Y, Z)。 |
| | `draw_result` | **擴展畫布設計**：繪製延伸線、VP、姿態文字。 |
| | `process_and_display` | 自動化測試流程控管。 |

```mermaid
graph TD
%% 核心節點定義
Main["main 主控流程"]
SubProcess["process_and_display 情境顯示控制"]

%% 第一層：主流程呼叫
Main --> READ["read_image 影像讀取"]
Main --> PRE["preprocess 影像前處理"]
Main --> HOUGH["detect_hough_lines 直線偵測"]
Main --> MERGE["merge_similar_lines 線段合併"]
Main --> SubProcess

%% 線段優化群組
subgraph Line_Optimization [線段優化模組]
    MERGE --> S2S["distance_segment_to_segment"]
    S2S --> P2S["distance_point_to_segment"]
end

%% 姿態解算群組
subgraph Execution_and_View [姿態解算與視覺化]
    SubProcess --> VP["find_vanishing_point"]
    SubProcess --> EYP["estimate_yaw_pitch"]
    SubProcess --> ER["estimate_roll"]
    SubProcess --> DRAW["draw_result 視覺化引擎"]
end

%% 基礎幾何工具
subgraph Geometry_Utilities [幾何工具組]
    VP --> INTER["intersection 直線交點"]
    VP --> DIST["dist_points 兩點距離"]
    DRAW --> BOUND["get_line_boundary_points"]
    DRAW --> AXIS["draw_camera_axis 座標軸繪製"]
end

%% 樣式設定
style Main fill:#f9f,stroke:#333,stroke-width:2px
style SubProcess fill:#bbf,stroke:#333,stroke-width:2px
style DRAW fill:#dfd,stroke:#333
```

    
### 3.3 API 定義 (主要函式參數)

| 函式名稱 | 輸入參數 | 輸出結果 |
| :--- | :--- | :--- |
| `merge_similar_lines` | `lines`, `angle_th`, `dist_th` | 合併後的精簡線段列表 |
| `find_vanishing_point` | `lines`, `width`, `height` | 消失點座標 (x, y) |
| `get_line_boundary_points`| `A, B, C` (直線係數), `w, h` | 畫布邊界的兩個端點 |
| `draw_result` | `img`, `lines`, `vp`, `yaw`, `pitch`, `roll` | **擴張後的** 結果圖像 |

---

### 3.3 API 定義 (完整版)

#### **A. 基礎 IO 與影像前處理**
| 函式名稱 | 輸入參數 | 輸出結果 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `read_image` | `path` (str) | `img` (ndarray) | 讀取影像。支援繁體中文路徑與自動轉換 Alpha 通道至 BGR 格式。 |
| `preprocess` | `img` (ndarray) | `resized, gray, edges` | 影像前處理：Resize 至 640x480、轉灰階、高斯模糊、以及**自適應 Canny 邊緣偵測**。 |

#### **B. 幾何運算輔助模組 (Geometry Helpers)**
| 函式名稱 | 輸入參數 | 輸出結果 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `dist_points` | `p1, p2` (tuple) | `distance` (float) | 計算平面兩點間的歐幾里德距離。 |
| `distance_point_to_segment` | `pt, start, end` | `distance` (float) | 計算點到指定線段的最短距離（垂直距離或到端點距離）。 |
| `distance_segment_to_segment` | `line1, line2` | `distance` (float) | **關鍵運算**：計算空間中兩條線段之間的最短距離，用於線段合併判斷。 |
| `intersection` | `line1, line2` | `(x, y)` or `None` | 計算兩條直線的交點；若兩線平行則回傳 `None`。 |
| `get_line_boundary_points` | `A, B, C, w, h` | `list of points` | 給定直線方程式 $Ax+By+C=0$，計算其與畫布邊界（含擴展區域）的交點。 |

#### **C. 核心特徵提取與合併模組**
| 函式名稱 | 輸入參數 | 輸出結果 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `detect_hough_lines` | `edges` (ndarray) | `lines` (list) | 執行機率 Hough 轉換，並過濾過短線段。回傳包含 `(x1, y1, x2, y2, length, angle)` 的資訊。 |
| `merge_similar_lines` | `lines, angle_th, dist_th` | `merged_lines` | **核心邏輯**：將角度相近且距離鄰近的碎裂線段進行聚類，僅保留各組中最長的特徵線。 |
| `find_vanishing_point` | `lines, w, h` | `(vp_x, vp_y)` | **聚類演算法**：尋找所有線段交點中，擁有最多「唯一線段支持」的聚類中心（中位數），定位消失點。 |

#### **D. 相機姿態估算模組**
| 函式名稱 | 輸入參數 | 輸出結果 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `estimate_roll` | `lines` (list) | `roll` (float) | 提取接近水平方向的線段群組，以其中位數角度估算相機 Roll。 |
| `estimate_yaw_pitch` | `vp, w, h` | `yaw, pitch` | 根據消失點相對於主點的偏移量，配合假設焦距 $f$ 反算 Yaw 與 Pitch 角度。 |

#### **E. 視覺化引擎與流程主控**
| 函式名稱 | 輸入參數 | 輸出結果 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `draw_camera_axis` | `img, vp, roll, offset` | `(None)` | 在影像中心（或偏移中心）繪製 3D 相機座標軸 (X-紅, Y-綠, Z-藍)。 |
| `draw_result` | `img, lines, vp, yaw, pitch, roll, ...` | `out_img` (ndarray) | **繪圖核心**：支援**自動畫布擴張**、繪製延伸輔助線、標註消失點與姿態文字。 |
| `process_and_display` | `img, lines, name, ...` | `(None)` | 封裝單次處理流程，包含計算、視覺化、Colab 顯示與檔案儲存。 |
| `main` | (無) | (無) | 主程式：定義路徑、呼叫流程，並執行「未合併」與「合併後」的多情境對照實驗。 |

---

## 4. 程式設計 Coding

### 4.1 主要技術與套件
- **Python 3.x**
- **OpenCV (cv2)**：核心影像處理庫。
- **NumPy (np)**：矩陣運算與中位數統計。
- **Math**：三角函數計算。
- **OS**：檔案路徑與目錄管理。
- **google.colab.patches**：Colab 顯示補丁。

### 4.2 執行方式 (How to Run)
1. **環境準備**：
   ```bash
   pip install opencv-python numpy
   ```
2. **路徑設定**：於 `main()` 中修改 `img_path` 指向您的測試圖片。
3. **執行**：執行後會自動產生「合併前/合併後」及「含延伸線/不含延伸線」的對照組，幫助驗證演算法穩定性。

---

## 5. 驗證 Verification

### 5.1 驗證指標
- **線段去雜訊效果**：觀察 `With Merging` 版本是否能排除過多細碎交點。
- **畫布擴展精度**：檢查延伸藍線是否確實交會於紅色的消失點。
- **姿態數值穩定度**：在相似場景下，Roll 誤差是否在 ±5 度內。

### 5.2 預期輸出
- 同時產出四種情境圖儲存於 `/picture/` 資料夾下：
    1. `Without Merging`: 原始線段偵測。
    2. `With Merging`: 經過精簡後的特徵線。
    3. 各版本的 `No Extended`: 僅顯示原圖內線段。

---

## 6. 結論 

本專案已能在不同 `minLineLength` 設定下輸出姿態估測結果。以下以同一張測試圖為例，對照 `minLineLength=50` 與 `minLineLength=100` 的主要輸出差異。

| 圖片類型 | minLineLength=50 | minLineLength=100 |
| :--- | :--- | :--- |
| 原圖 | ![test_picture_1](picture/minLineLength=50/test_picture_1.png) | ![test_picture_1](picture/minLineLength=100/test_picture_1.png) |
| 合併後結果 | ![merged](picture/minLineLength=50/test_picture_1_camera_pose_result_merged.png) | ![merged](picture/minLineLength=100/test_picture_1_camera_pose_result_merged.png) |
| 未合併結果 | ![raw](picture/minLineLength=50/test_picture_1_camera_pose_result_raw.png) | ![raw](picture/minLineLength=100/test_picture_1_camera_pose_result_raw.png) |
| 合併後（No Extended） | ![merged no extended](picture/minLineLength=50/test_picture_1_camera_pose_result_merged_no_extended.png) | ![merged no extended](picture/minLineLength=100/test_picture_1_camera_pose_result_merged_no_extended.png) |
| 未合併（No Extended） | ![raw no extended](picture/minLineLength=50/test_picture_1_camera_pose_result_raw_no_extended.png) | ![raw no extended](picture/minLineLength=100/test_picture_1_camera_pose_result_raw_no_extended.png) |
| 原圖 | ![test_picture_2](picture/minLineLength=50/test_picture_2.png) | ![test_picture_2](picture/minLineLength=100/test_picture_2.png) |
| 合併後結果 | ![merged](picture/minLineLength=50/test_picture_2_camera_pose_result_merged.png) | ![merged](picture/minLineLength=100/test_picture_2_camera_pose_result_merged.png) |
| 未合併結果 | ![raw](picture/minLineLength=50/test_picture_2_camera_pose_result_raw.png) | ![raw](picture/minLineLength=100/test_picture_2_camera_pose_result_raw.png) |
| 合併後（No Extended） | ![merged no extended](picture/minLineLength=50/test_picture_2_camera_pose_result_merged_no_extended.png) | ![merged no extended](picture/minLineLength=100/test_picture_2_camera_pose_result_merged_no_extended.png) |
| 未合併（No Extended） | ![raw no extended](picture/minLineLength=50/test_picture_2_camera_pose_result_raw_no_extended.png) | ![raw no extended](picture/minLineLength=100/test_picture_2_camera_pose_result_raw_no_extended.png) |
| 原圖 | ![test_picture_3](picture/minLineLength=50/test_picture_3.png) | ![test_picture_3](picture/minLineLength=100/test_picture_3.png) |
| 合併後結果 | ![merged](picture/minLineLength=50/test_picture_3_camera_pose_result_merged.png) | ![merged](picture/minLineLength=100/test_picture_3_camera_pose_result_merged.png) |
| 未合併結果 | ![raw](picture/minLineLength=50/test_picture_3_camera_pose_result_raw.png) | ![raw](picture/minLineLength=100/test_picture_3_camera_pose_result_raw.png) |
| 合併後（No Extended） | ![merged no extended](picture/minLineLength=50/test_picture_3_camera_pose_result_merged_no_extended.png) | ![merged no extended](picture/minLineLength=100/test_picture_3_camera_pose_result_merged_no_extended.png) |
| 未合併（No Extended） | ![raw no extended](picture/minLineLength=50/test_picture_3_camera_pose_result_raw_no_extended.png) | ![raw no extended](picture/minLineLength=100/test_picture_3_camera_pose_result_raw_no_extended.png) |
| 原圖 | ![test_picture_4](picture/minLineLength=50/test_picture_4.png) | ![test_picture_4](picture/minLineLength=100/test_picture_4.png) |
| 合併後結果 | ![merged](picture/minLineLength=50/test_picture_4_camera_pose_result_merged.png) | ![merged](picture/minLineLength=100/test_picture_4_camera_pose_result_merged.png) |
| 未合併結果 | ![raw](picture/minLineLength=50/test_picture_4_camera_pose_result_raw.png) | ![raw](picture/minLineLength=100/test_picture_4_camera_pose_result_raw.png) |
| 合併後（No Extended） | ![merged no extended](picture/minLineLength=50/test_picture_4_camera_pose_result_merged_no_extended.png) | ![merged no extended](picture/minLineLength=100/test_picture_4_camera_pose_result_merged_no_extended.png) |
| 未合併（No Extended） | ![raw no extended](picture/minLineLength=50/test_picture_4_camera_pose_result_raw_no_extended.png) | ![raw no extended](picture/minLineLength=100/test_picture_4_camera_pose_result_raw_no_extended.png) |

從表格可以看出，`minLineLength=50` 保留較多短線段，因此交點與延伸線更密集；`minLineLength=100` 則會過濾掉較短雜訊線，使畫面更乾淨，但也可能少掉部分細節線段。實際使用時可依場景複雜度在兩者之間取捨。

---

## 7. 主要參數調校 (How to Tune)
- **`angle_threshold`**: 若場景線段雜亂，可調小此值（如 5）使合併更嚴格。
- **`dist_threshold`**: 若線段較碎裂，可調大此值（如 50）強化合併效果。
- **`fx, fy`**: 相機焦距參數，若姿態角度偏差過大，請根據相機規格調整此值。