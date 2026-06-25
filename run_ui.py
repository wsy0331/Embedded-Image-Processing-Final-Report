import subprocess
import sys
import tkinter as tk
from tkinter import filedialog
import os

def select_video():
    """開啟檔案選擇視窗，讓使用者選擇影片，並直接定位在同個資料夾"""
    # 建立一個隱藏的 tkinter 主視窗
    root = tk.Tk()
    root.withdraw() 
    
    # 自動獲取目前 B.py 所在的絕對路徑資料夾
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 彈出選擇檔案視窗，加入 initialdir 參數
    file_path = filedialog.askopenfilename(
        title="請選擇要測試的影片",
        initialdir=current_dir,  
        filetypes=[
            ("影片檔案", "*.mp4 *.avi *.mkv *.mov"), 
            ("所有檔案", "*.*")
        ]
    )
    return file_path

def run_scripts():
    print("🚀 準備啟動影像處理測試...")
    
    # 1. B.py 本身執行：讓使用者選擇影片
    video_path = select_video()
    
    # 如果使用者按了取消，沒有選擇檔案，則結束程式
    if not video_path:
        print("❌ 未選擇影片，程式結束。")
        return
        
    print(f"📁 已選擇影片: {video_path}")
    
    python_exe = sys.executable

    # 2. 🌟【修改之處】將原本的 main.py 改為啟動 A.py
    # 透過 Popen 非阻塞特性，同時並行啟動 A.py 與 p.py
    print("▶️ 正在啟動 A.py (多執行緒/雙核光流/時間加總版本)...")
    process_A = subprocess.Popen([python_exe, "A.py", video_path])

    print("▶️ 正在啟動 p.py (極端單核心鎖定版本)...")
    process_p = subprocess.Popen([python_exe, "p.py", video_path])

    print("\n✅ 所有程式已成功啟動！(總管 B.py 正在幕後監控)")
    print("💡 提示：請將跳出的兩個影片視窗拖開，即可並排對比單核與多核的 Process Time。")
    print("⏳ 等待使用者關閉視窗或程式執行完畢...\n")

    # 3. 🌟【修改之處】讓 B.py 在後台守候這兩個子程序的結束
    process_A.wait()
    process_p.wait()
    
    print("🏁 所有測試皆已結束，總管程式 B.py 安全關閉。")

if __name__ == "__main__":
    run_scripts()