import streamlit as st
import requests
import base64
import os
import json
import time
import random
import cv2
import numpy as np
import re
import pandas as pd
import shutil
from collections import Counter
from bs4 import BeautifulSoup
from openpyxl import Workbook, load_workbook
from io import BytesIO

# ==================== 1. 你的原始核心逻辑 ====================

def preprocess_image(image_path, mode='light'):
    img = cv2.imread(image_path)
    if img is None: return image_path
    height, width = img.shape[:2]
    min_dim = min(height, width)
    if min_dim < 800:
        scale = 1200 / min_dim
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0 if mode=='light' else 3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    temp_path = f"pre_{mode}_{os.path.basename(image_path)}"
    cv2.imwrite(temp_path, gray)
    return temp_path

def api_call_with_retry(api_url, token, file_data, max_retries=5):
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    payload = {"file": file_data, "fileType": 1, "useDocOrientationClassify": False, "useDocUnwarping": False, "useChartRecognition": False}
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            if response.status_code == 200: return response.json(), None
            if response.status_code == 503:
                time.sleep(min((2 ** attempt), 30))
                continue
            return None, f"API失败: {response.status_code}"
        except Exception as e:
            time.sleep(2)
    return None, "超过最大重试次数"

def call_paddleocr_api(image_path, api_url, token, preprocess_mode=None):
    temp_path = None
    try:
        target_path = preprocess_image(image_path, preprocess_mode) if preprocess_mode else image_path
        with open(target_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("ascii")
        result, error = api_call_with_retry(api_url, token, file_data)
        if error: return None, error
        md_text = result["result"]["layoutParsingResults"][0]["markdown"]["text"]
        return md_text, None
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

def parse_html_to_2d_list(html_str):
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table')
        if not table: return None
        rows_data = []
        occupied = set()
        for tr in table.find_all('tr'):
            row_cells = []; col_idx = 0
            for cell in tr.find_all(['td', 'th']):
                while (len(rows_data), col_idx) in occupied:
                    row_cells.append(""); col_idx += 1
                text = cell.get_text(strip=True)
                row_cells.append(text)
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                if rowspan > 1 or colspan > 1:
                    for r in range(rowspan):
                        for c in range(colspan):
                            if r == 0 and c == 0: continue
                            occupied.add((len(rows_data) + r, col_idx + c))
                col_idx += colspan
            rows_data.append(row_cells)
        return rows_data
    except: return None

def merge_multiple_results(results_list):
    if not results_list: return []
    base_rows = len(results_list[0])
    base_cols = max(len(row) for row in results_list[0])
    merged = []
    for r in range(base_rows):
        row_data = []
        for c in range(base_cols):
            candidates = [res[r][c].strip() for res in results_list if r < len(res) and c < len(res[r]) and res[r][c].strip()]
            row_data.append(Counter(candidates).most_common(1)[0][0] if candidates else "")
        merged.append(row_data)
    return merged

def apply_workshop_logic(table_data, workshop):
    if workshop == "下货车间" and table_data:
        keywords = {"weight": ["重", "重量"], "prev": ["上日"], "current": ["今日", "当日"]}
        found_cols = {k: None for k in keywords}
        for r_idx in range(min(5, len(table_data))):
            for c_idx, cell in enumerate(table_data[r_idx]):
                clean = re.sub(r'[^\w]', '', str(cell))
                for k, v in keywords.items():
                    if any(kw in clean for kw in v): found_cols[k] = c_idx
        if all(v is not None for v in found_cols.values()):
            table_data[0].append("合计")
            for row in table_data[1:]:
                def get_num(idx):
                    val = row[idx] if idx < len(row) else "0"
                    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
                    return float(m.group(1)) if m else 0.0
                total = get_num(found_cols["weight"]) + get_num(found_cols["current"]) - get_num(found_cols["prev"])
                row.append(str(total))
    return table_data

# ==================== 2. Streamlit UI 界面 ====================

st.set_page_config(page_title="AI 报表自动化", layout="wide")

st.title("🏭 智能车间报表 OCR 汇总工具")
st.info("说明：上传图片后，系统将自动识别表格并根据车间逻辑计算，最后生成汇总 Excel。")

# 侧边栏配置
with st.sidebar:
    st.header("配置选项")
    api_url = st.text_input("OCR API 地址", value="http://your-api-url/predict/layout")
    token = st.text_input("API Token", type="password")
    workshop = st.selectbox("所属车间", ["下货车间", "分割车间"])
    mode = st.select_slider("图像增强模式", options=["none", "light", "medium", "strong"], value="light")
    repeat = st.number_input("重复识别次数（提高准确率）", 1, 5, 1)

uploaded_files = st.file_uploader("上传报表图片", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and api_url and token:
    if st.button("开始处理并生成汇总表"):
        all_results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # 临时保存
            t_path = f"temp_{file.name}"
            with open(t_path, "wb") as f:
                f.write(file.getvalue())
            
            st.write(f"正在处理: {file.name}...")
            
            # 执行识别逻辑
            tables = []
            for _ in range(repeat):
                html, err = call_paddleocr_api(t_path, api_url, token, None if mode=="none" else mode)
                if not err:
                    table = parse_html_to_2d_list(html)
                    if table: tables.append(table)
            
            if tables:
                final_table = merge_multiple_results(tables) if len(tables)>1 else tables[0]
                final_table = apply_workshop_logic(final_table, workshop)
                all_results.append(final_table)
                st.success(f"✅ {file.name} 识别完成")
            else:
                st.error(f"❌ {file.name} 识别失败")
            
            os.remove(t_path)
            progress_bar.progress((i + 1) / len(uploaded_files))

        if all_results:
            st.divider()
            st.subheader("📋 处理结果汇总")
            
            # 合并所有结果到一个 DataFrame 预览
            combined_df = []
            for t in all_results:
                combined_df.append(pd.DataFrame(t[1:], columns=t[0]))
            
            final_df = pd.concat(combined_df, ignore_index=True)
            st.dataframe(final_df, use_container_width=True)

            # 导出 Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False, sheet_name='识别结果')
            
            st.download_button(
                label="📥 下载生成的 Excel 报表",
                data=output.getvalue(),
                file_name=f"{workshop}_汇总_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.warning("请在左侧输入 API 信息并上传图片。")
