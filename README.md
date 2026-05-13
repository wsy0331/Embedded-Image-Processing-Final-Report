# Embedded-Image-Processing-Final-Report
嵌入式影像處裡的期末報告

---

# 需求規格分析書：火災與煙霧偵測系統

## 1. 專案概述
本專案旨在開發一個能即時偵測影片中「火焰」與「煙霧」的視覺系統。透過結合運動偵測（光流法）與顏色特徵分析，系統需標記目標位置並輸出邊界框（Bounding Box），同時針對 Raspberry Pi 4B 等嵌入式硬體進行效能優化。

## 2. 功能需求 (Functional Requirements)

### 2.1 影像輸入與預處理
*   **影像來源**：支援影片檔案（如 `.mp4`）或即時攝影機串流。
*   **解析度調整**：系統應將輸入影像縮放至指定尺寸（如 1280x720 或更低，以適應 RPi 效能），確保處理速度。
*   **灰階轉換**：用於運動分析（Farneback 光流法）。

### 2.2 火焰/煙霧特徵檢測
*   **運動區域檢測 (Motion Detection)**：
    *   使用 **Farneback 稠密光流法** 計算相鄰幀之間的位移。
    *   過濾微小運動（雜訊），提取具備明顯動態特徵的區域。
*   **顏色特徵分析 (Color Analysis)**：
    *   **火焰**：基於 RGB 通道比例（R > G > B 且 $R > \text{threshold}$），利用 $R/G$ 與 $R/B$ 的比例過濾紅色/橙色區域。
    *   **煙霧（擴充需求）**：偵測高飽和度低、色彩偏灰且具備擴散特徵的區域。
*   **邏輯融合**：結合運動遮罩與顏色遮罩（`bitwise_and`），減少因環境光影（如紅色路燈、夕陽）造成的誤報。

### 2.3 目標定位與輸出
*   **輪廓提取**：對融合後的二值化遮罩進行形態學處理（閉運算與開運算），去除雜訊並填補孔洞。
*   **邊界標記**：
    *   計算目標面積（Area）。
    *   產出 **Bounding Box** 或輪廓描邊。
    *   標註偵測到的火源數量與運動像素統計。

---

## 3. 進階需求

### 3.1 硬體與效能 (Bonus 項目相關)
*   **硬體限制**：目標運行於 Raspberry Pi 4B (Broadcom BCM2711, 4-core Cortex-A72)。
*   **FPS 優化**：
    *   在 720p 解析度下，Farneback 光流法運算量極大。需考量縮小計算解析度（如 640x360）或調整 `pyr_scale` 與 `iterations` 參數。
    *   目標 FPS：應維持在 10-15 FPS 以上以達到實時監控感。
*   **環境適應性**：
    *   系統需能應對不同光影變化（透過動態調整比例閾值或使用 HSV 色彩空間替代單純 RGB）。

### 3.2 穩定性
*   系統需具備錯誤處理機制（如影片讀取結束自動釋放資源）。

---

## 4. 系統限制與挑戰

1.  **光流計算壓力**：Farneback 算法對 RPi 4B 負擔重。
    *   *對策*：可考慮改用 `calcOpticalFlowPyrLK`（稀疏光流）或簡單的「背景減除法 (MOG2)」來替代部分場景的偵測。
2.  **煙霧偵測難度**：煙霧顏色不固定且邊界模糊。
    *   *對策*：加強對形態學演變（體積增大特徵）的監測。
3.  **誤報干擾**：環境中若有與火焰顏色相近的移動物體（如穿紅衣的人走過）。
    *   *對策*：引入時間連續性判定，火焰通常伴隨高頻閃爍或不規則形狀變化。

---

## 5. 輸出規格 (Output Specification)

| 項目 | 描述 | 範例 |
| :--- | :--- | :--- |
| **視覺輸出** | 原始影像 + 紅色輪廓/邊界框 | `cv2.drawContours` / `cv2.rectangle` |
| **數據標籤** | 顯示偵測數量、各區塊面積 | `Fire Sources: 2`, `Area: 1250` |
| **性能統計** | 當前處理解析度與運算強度 | `1280x720`, `Motion: 5000 pixels` |

---

## 6. 專案執行計畫 (針對期末)

1.  **第一階段**：完善 `test.py` 中的顏色特徵公式，加入煙霧偵測邏輯。
2.  **第二階段**：在 Raspberry Pi 4B 上測試，記錄不同解析度（720p vs 360p）下的 FPS 差異。
3.  **第三階段**：進行環境光測試（開燈/關燈/不同色溫），調整比例參數以提升魯棒性。
4.  **第四階段**：輸出符合要求的 Bounding Box 與最終專案報告。
---
https://github.com/wsy0331/Embedded-Image-Processing-Final-Report/blob/main/output_videos/fire_detection_output.mp4
## 7. 目前效果

<video id="video" controls="" preload="none" poster="封面">
      <source id="mp4" src="https://github.com/wsy0331/Embedded-Image-Processing-Final-Report/blob/main/output_videos/fire_detection_output.mp4" type="video/mp4">
</videos>
