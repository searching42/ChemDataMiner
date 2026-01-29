import glob
import pandas as pd
from io import StringIO
from tqdm import tqdm
from bs4 import BeautifulSoup
from src.config import RAW_DIR, FILES, GARBAGE_ROW_STARTS
from src.utils import clean_header_text

def extract_tables_from_md(file_path):
    extracted = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            tables = soup.find_all('table')
            for idx, table in enumerate(tables):
                df_list = pd.read_html(str(table))
                if df_list:
                    extracted.append(df_list[0])
    except Exception:
        pass
    return extracted

def clean_dataframe(df):
    """核心清洗逻辑"""
    # 1. 如果全是数字表头，修正表头
    try:
        cols = [str(c) for c in df.columns]
        if sum(1 for c in cols if c.isdigit() or "unnamed" in c.lower()) / len(cols) > 0.8:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
    except:
        pass

    if df.empty or df.shape[1] < 2: return None

    # 2. 清洗列名
    df.columns = [clean_header_text(c) for c in df.columns]
    
    # 3. 剔除垃圾行
    df = df[~df.iloc[:, 0].astype(str).str.strip().isin(GARBAGE_ROW_STARTS)]
    
    # 4. 剔除原子坐标表等非目标表 (简单规则：第一列包含太多元素符号)
    # 这里为了简化，仅做基础过滤
    
    return df.dropna(how='all')

def run():
    print(">>> STEP 1: Extracting and Cleaning Tables...")
    # 支持递归查找
    md_files = glob.glob(str(RAW_DIR / "**/*.md"), recursive=True)
    print(f"Found {len(md_files)} markdown files.")
    
    valid_tables = []
    
    for fpath in tqdm(md_files):
        tables = extract_tables_from_md(fpath)
        for i, df in enumerate(tables):
            clean_df = clean_dataframe(df)
            if clean_df is not None:
                # 转为 Markdown Grid 格式字符串
                table_str = clean_df.to_markdown(index=False, tablefmt="grid")
                valid_tables.append(
                    f"FILE: {fpath}\nTABLE_ID: {i+1}\n{table_str}\n======\n"
                )

    with open(FILES['tables_final'], 'w', encoding='utf-8') as f:
        f.writelines(valid_tables)
    
    print(f"Saved {len(valid_tables)} tables to {FILES['tables_final']}")

if __name__ == "__main__":
    run()