# 請先在終端機執行：
# pip install streamlit ultralytics opencv-python-headless pillow

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO

# --- 1. 設定頁面配置 ---
st.set_page_config(page_title="皮膚偵測 AI 系統", layout="wide")
st.title("🔍 皮膚偵測與分析系統")
st.write("上傳圖片並調整亮度，即可進行即時 AI 偵測")

# --- 2. 載入模型 (快取處理) ---
@st.cache_resource
def load_model():
    # 請確保 best.pt 與此程式碼在同一資料夾下
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"模型載入失敗，請確認檔案名稱是否為 best.pt: {e}")
        return None

model = load_model()

# --- 3. 側邊欄設定 ---
st.sidebar.header("參數設定")
brightness = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
conf_threshold = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 4. 圖片上傳區域 ---
uploaded_file = st.file_uploader("請選擇一張皮膚照片 (jpg, png, jpeg)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取並處理亮度
    image = Image.open(uploaded_file)
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(brightness)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 待測圖片")
        st.image(processed_image, use_container_width=True)
    
    # 準備偵測
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if st.button("🚀 開始 AI 偵測"):
        if model is None:
            st.error("模型未載入，無法執行偵測。")
        else:
            with st.spinner('AI 正在分析中...'):
                results = model.predict(source=img_bgr, conf=conf_threshold)
                
                # 取得畫好框的圖片
                annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("✅ 偵測結果")
                    st.image(annotated_img, use_container_width=True)
                
                # --- 核心修正：顯示下方詳細資訊 ---
                st.markdown("---")
                st.subheader("📊 偵測數據詳情")
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"偵測完成！共發現 {len(boxes)} 處目標。")
                    
                    # 使用欄位顯示表頭
                    h1, h2, h3 = st.columns([1, 2, 2])
                    h1.write("**序號**")
                    h2.write("**類別名稱**")
                    h3.write("**信心指數**")
                    
                    # 迴圈讀取每一個偵測到的物件
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])           # 取得類別 ID
                        label = model.names[class_id]        # 轉換為名稱
                        confidence = float(box.conf[0])      # 取得信心值
                        
                        # 顯示每一列內容
                        r1, r2, r3 = st.columns([1, 2, 2])
                        r1.write(f"{i+1}")
                        r2.info(f"**{label}**")
                        r3.write(f"{confidence:.2%}") # 顯示百分比格式
                else:
                    st.warning("未偵測到任何目標，請嘗試降低「信心度門檻」或調整「亮度」。")

# 執行指令: streamlit run main.py