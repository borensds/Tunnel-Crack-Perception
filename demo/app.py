import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys
import zipfile
import traceback
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# 🚨 动态挂载，确保能读取上层目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

try:
    from core.postprocess import FractureAnalyzer
    from tools.inference import OptimizedInferenceEngine
except ImportError as e:
    st.error(f"核心模块导入失败: {e}")
    st.info("请检查 core/ 和 tools/ 目录结构是否完整。")
    st.stop()

st.set_page_config(layout="wide", page_title="隧道智能感知系统 PoC")


class ImageCalibrator:
    @staticmethod
    def perform_perspective_correction(img, pts, real_w, real_h):
        pts = np.float32(pts)
        pts_dst = np.float32([[0, 0], [real_w, 0], [real_w, real_h], [0, real_h]])
        matrix = cv2.getPerspectiveTransform(pts, pts_dst)
        warped = cv2.warpPerspective(img, matrix, (int(real_w), int(real_h)))
        # 计算比例尺 mm/px (假设变换后 1 像素 = 1 mm，因为我们传参就是 real_w)
        # 这里仅为 PoC 演示，实际中需根据标定板的物理尺寸与像素比精确换算
        return warped, 1.0


if 'canvas_key' not in st.session_state: st.session_state['canvas_key'] = 0
if 'scale' not in st.session_state: st.session_state['scale'] = None
if 'warped_img' not in st.session_state: st.session_state['warped_img'] = None
if 'mask_crack' not in st.session_state: st.session_state['mask_crack'] = None
if 'mask_face' not in st.session_state: st.session_state['mask_face'] = None
if 'df_results' not in st.session_state: st.session_state['df_results'] = None
if 'last_uploaded_name' not in st.session_state: st.session_state['last_uploaded_name'] = ""
if 'roi_coords' not in st.session_state: st.session_state['roi_coords'] = None
if 'final_cal_points' not in st.session_state: st.session_state['final_cal_points'] = []


@st.cache_resource
def load_ai_engine():
    weight_path = os.path.join(ROOT_DIR, "weights", "best_ours.pth")
    if not os.path.exists(weight_path):
        return None
    try:
        return OptimizedInferenceEngine(crack_model_path=weight_path)
    except Exception as e:
        st.error(f"模型引擎初始化失败: {e}")
        return None


ai_engine = load_ai_engine()


def draw_overlay(base_img, mask_face, mask_crack, df, show_face, show_crack, show_id):
    vis_img = base_img.copy()
    h, w = base_img.shape[:2]
    font_scale = max(0.4, w / 2500.0)
    thickness = max(1, int(w / 1500.0))

    if show_face and mask_face is not None:
        try:
            contours, _ = cv2.findContours(mask_face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (0, 0, 255), thickness + 1)
        except:
            pass

    if show_crack and mask_crack is not None:
        try:
            red_mask = np.zeros_like(vis_img)
            red_mask[mask_crack > 0] = [255, 0, 0]
            mask_indices = mask_crack > 0
            vis_img[mask_indices] = cv2.addWeighted(vis_img[mask_indices], 0.6, red_mask[mask_indices], 0.4, 0)
        except:
            pass

    if show_id and df is not None and not df.empty:
        overlay = vis_img.copy()
        for idx, row in df.iterrows():
            if pd.notnull(row.get('中心X')) and pd.notnull(row.get('中心Y')):
                cx, cy = int(row['中心X']), int(row['中心Y'])
                cid = str(int(row['ID']))
                (tw, th), _ = cv2.getTextSize(cid, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(overlay, (cx, cy - th - 2), (cx + tw, cy + 2), (255, 255, 255), -1)

        cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)

        for idx, row in df.iterrows():
            if pd.notnull(row.get('中心X')) and pd.notnull(row.get('中心Y')):
                cx, cy = int(row['中心X']), int(row['中心Y'])
                cid = str(int(row['ID']))
                cv2.putText(vis_img, cid, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 139), thickness)
    return vis_img


with st.sidebar:
    st.title("隧道分析控制台")
    if ai_engine is None:
        st.error("未检测到模型权重，系统运行在纯后处理模式。请先执行 python tools/train.py")
    else:
        st.success("MDC-Net 模型加载就绪")

    st.header("拓扑参数设置")
    min_area = st.slider("去噪阈值 (Min Seg Pts)", 0, 50, 6)
    connect_gap = st.slider("裂隙缝合力度 (Connect Gap)", 0, 30, 12)

st.title("隧道掌子面裂隙智能感知与量化系统 (PoC)")
tab1, tab2 = st.tabs(["AI 视觉感知", "工程量化报告"])

with tab1:
    uploaded_file = st.file_uploader("上传掌子面影像", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        raw_image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)

        if st.session_state['last_uploaded_name'] != uploaded_file.name:
            st.session_state['warped_img'] = raw_image
            st.session_state['last_uploaded_name'] = uploaded_file.name

        st.image(raw_image, caption="原始影像", use_column_width=True)

        if st.button("执行全景网络感知", type="primary"):
            if not ai_engine:
                st.error("没有模型权重，无法进行 AI 推理！")
                st.stop()

            with st.status("正在启动边缘计算流水线...", expanded=True) as status_bar:
                try:
                    status_bar.write("1/3: 启动张量加速推理...")
                    c_mask, f_mask = ai_engine.predict_full_pipeline(raw_image)

                    status_bar.write("2/3: 执行 C-HTP 拓扑解耦...")
                    # 模拟组件 Label map (这里简化处理，实际中可能需要聚类或分块算法赋 ID)
                    num_labels, label_map = cv2.connectedComponents(c_mask)

                    status_bar.write("3/3: 亚像素级几何量化...")
                    analyzer = FractureAnalyzer(
                        pixel_scale=1.0,
                        min_seg_pts=min_area,
                        connect_gap=connect_gap
                    )

                    df = analyzer.analyze_with_labels(c_mask, label_map, num_labels)

                    st.session_state['mask_crack'] = c_mask
                    st.session_state['mask_face'] = f_mask
                    st.session_state['df_results'] = df
                    status_bar.update(label="分析完成！", state="complete", expanded=False)

                    st.success("计算完成，请切换至 [工程量化报告] 标签查看数据。")
                except Exception as e:
                    status_bar.update(label="推理崩溃", state="error")
                    st.error(f"错误堆栈: {traceback.format_exc()}")

with tab2:
    if st.session_state['df_results'] is not None:
        df = st.session_state['df_results']

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### 裂隙特征清单")
            st.dataframe(df, use_container_width=True, height=400)

        with c2:
            st.markdown("### 定量映射图")
            base = st.session_state['warped_img']
            mc = st.session_state['mask_crack']
            img_final_ov = draw_overlay(base, None, mc, df, False, True, True)
            st.image(img_final_ov, caption="带有独立 ID 编号的拓扑还原图", use_column_width=True)