import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def main():
    # --- 1. 介面設定 ---
    st.set_page_config(page_title="影像特效實驗室 Pro", layout="wide")
    
    # --------------------------------------------------------
    # 👇👇👇 這裡就是修改後的部分：使用 HTML 讓標題變大且置中 👇👇👇
    st.markdown(
        """
        <style>
        .title-style {
            font-size: 50px;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle-style {
            font-size: 20px;
            color: #7F8C8D;
            text-align: center;
            margin-bottom: 30px;
        }
        </style>
        <div class="title-style"> 影像處理期末報告：進階特效 APP</div>
        <div class="subtitle-style">組員：林于喬  |  呂威漢  |  陳翊中  |  林定緯</div>
        """,
        unsafe_allow_html=True
    )
    # --------------------------------------------------------

    # --- 2. 側邊欄：設定區 ---
    with st.sidebar:
        st.header("1. 上傳圖片")
        uploaded_file = st.file_uploader("請選擇圖片", type=['jpg', 'png', 'jpeg'])

        # 加入全域的亮度與對比調整
        st.header("2. 基礎調整")
        brightness = st.slider("亮度 (Brightness)", -100, 100, 0)
        contrast = st.slider("對比度 (Contrast)", -100, 100, 0)

        st.header("3. 選擇進階濾鏡")
        filter_type = st.selectbox(
            "特效模式",
            ["原圖", "素描風 ", "人臉偵測 ", "邊緣偵測 ", "馬賽克 "]
        )
        
        show_hist = st.checkbox("顯示直方圖分析 (Histogram)")

    # --- 3. 主程式邏輯 ---
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # A. 應用亮度與對比度
        alpha = (contrast + 100.0) / 100.0 
        beta = brightness
        img_adjusted = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)

        # B. 根據選擇的濾鏡進行處理
        final_img = img_adjusted.copy()

        if filter_type == "素描風 ":
            gray = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            final_img = cv2.divide(gray, 255 - blur, scale=256)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)

        elif filter_type == "邊緣偵測 ":
            t_lower = st.sidebar.slider("邊緣低閾值", 0, 255, 50)
            t_upper = st.sidebar.slider("邊緣高閾值", 0, 255, 150)
            final_img = cv2.Canny(final_img, t_lower, t_upper)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)

        elif filter_type == "人臉偵測 ":
            gray = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(final_img, (x, y), (x+w, y+h), (255, 0, 0), 4)
            st.sidebar.success(f"偵測到 {len(faces)} 張人臉！")

        elif filter_type == "馬賽克 ":
            level = st.sidebar.slider("馬賽克強度", 5, 50, 15)
            h, w, c = final_img.shape
            temp = cv2.resize(final_img, (w//level, h//level), interpolation=cv2.INTER_LINEAR)
            final_img = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- 4. 畫面排版顯示 ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("原始圖片 ")
            st.image(img_array, use_container_width=True) 
            
        with col2:
            st.subheader(f"處理結果: {filter_type}")
            st.image(final_img, use_container_width=True)

        # --- 5. 顯示直方圖 ---
        if show_hist:
            st.markdown("---")
            st.subheader("📊 影像直方圖分析 (Histogram Analysis)")
            st.caption("顯示 RGB 三原色的像素分佈，這是分析影像曝光與色彩的重要工具。")
            
            fig, ax = plt.subplots()
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img_adjusted], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
                ax.set_xlim([0, 256])
            
            st.pyplot(fig)

    else:
        # 這裡也幫你置中提示文字
        st.markdown("<h3 style='text-align: center; color: #999;'>👈 請從左側選單上傳圖片，開始你的影像魔法！</h3>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()