import os
import re
from pathlib import Path

# ================= 路径配置 =================
# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 数据根目录
DATA_DIR = BASE_DIR / "data"

# 1. 原始 PDF 输入目录 (Step 0 输入)
PDF_DIR = DATA_DIR / "00_pdfs"

# 2. Mineru 输出 / Markdown 原始目录 (Step 0 输出 / Step 1 输入)
# Mineru 会在这里生成文件夹，每个文件夹里有 .md 文件
RAW_DIR = DATA_DIR / "01_raw" 

# 3. 中间处理与结果目录
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_STAGING_DIR = DATA_DIR / "temp_staging" # Mineru 处理时的临时目录

# 自动创建必要目录
for p in [DATA_DIR, PDF_DIR, RAW_DIR, PROCESSED_DIR, TEMP_STAGING_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ================= 文件路径索引 =================
FILES = {
    # Step 1 输出：清洗合并后的表格文本
    "tables_final": PROCESSED_DIR / "tables_final.txt",

    # Step 1 输出（可选）：规范化后的表格/训练样本
    "tables_normalized": PROCESSED_DIR / "tables_normalized.jsonl",
    "tables_long": PROCESSED_DIR / "tables_long.jsonl",
    "step1_report": PROCESSED_DIR / "step1_report.json",
    
    # Step 2 输出：准备好喂给 LLM 的 JSONL
    "llm_input": PROCESSED_DIR / "llm_process.jsonl",
    
    # Step 3 输出：LLM 返回的原始结果
    "llm_output": PROCESSED_DIR / "post_llm.jsonl",
    
    # Step 4 输出：最终清洗好的数据集 CSV
    "dataset_csv": PROCESSED_DIR / "dataset.csv",
    
    # Step 4 输出：失败记录备份
    "failed_records": PROCESSED_DIR / "failed_records.jsonl"
}

# ================= 参数配置 =================

# Step 0: PDF 处理相关
MINERU_BATCH_SIZE = 100  # 每批处理多少个 PDF，防止内存溢出

# Step 1 & 4: 黑名单正则 (匹配到这些词则认为是无效数据)
BLACKLIST_PATTERNS = [
    r'^\d+$', r'^\d+[a-z]?$', r'^[a-z]\d+$', 
    r'nan', r'null', r'none', r'n/a',
    r'->', r'<-', r'⇌', 
    r'b3lyp', r'dft', r'hf', r'mp2', # 理论方法名不是分子名
    r'table', r'figure', r'scheme', r'entry', r'run', # 表头关键词
    r'temperature', r'solvent', r'yield', r'energy',
    r'calculated', r'experimental', r'value'
]

# Step 1: 垃圾行起始字符 (用于清洗表格)
GARBAGE_ROW_STARTS = ['\\', '-', '_', '+', '=', '!', '?', '|', '#']

# Step 4: 属性归一化映射 (将各种写法统一为标准字段)
PROPERTY_MAP = {
    # 能级
    "HOMO": "HOMO", "E_HOMO": "HOMO", "EHOMO": "HOMO", "IP": "HOMO", "HOMO energy": "HOMO",
    "LUMO": "LUMO", "E_LUMO": "LUMO", "ELUMO": "LUMO", "EA": "LUMO", "LUMO energy": "LUMO",
    "Gap": "Gap", "GAP": "Gap", "Eg": "Gap", "Band Gap": "Gap", "H-L Gap": "Gap", "Delta E": "Gap",
    # 产率
    "Yield": "Yield", "Chemical Yield": "Yield", "Isolated Yield": "Yield",
    # 光伏参数
    "PCE": "PCE", "Efficiency": "PCE",
    "Voc": "Voc", 
    "Jsc": "Jsc", 
    "FF": "FF",
    # 物理性质
    "Dipole": "Dipole_Moment", "Dipole Moment": "Dipole_Moment",
    "Solubility": "LogS", "LogS": "LogS",
    "LogP": "LogP", "cLogP": "LogP", "XlogP": "LogP",

    # --- Extended targets (for table normalization / model training) ---
    "IC50": "IC50", "IC 50": "IC50",
    "Dipole_Moment": "Dipole_Moment",
    "Isotropic Polarizability": "Isotropic_Polarizability",
    "Polarizability": "Isotropic_Polarizability",
    "Internal Energy": "Internal_Energy",
    "Enthalpy": "Enthalpy",
    "Gibbs Free Energy": "Gibbs_Free_Energy",
    "Hydration Free Energy": "Hydration_Free_Energy",
    "Toxicity": "Toxicity",
    "BBB": "BBB_Permeability", "BBB Permeability": "BBB_Permeability",
    "CYP450": "CYP450", "CYP 450": "CYP450",
    "Total Energy": "Total_Energy",
    "Atomic Forces": "Atomic_Forces",
    "Virial": "Virial_Stress_Tensor", "Stress Tensor": "Virial_Stress_Tensor",
    "Dielectric Constant": "Dielectric_Constant",
    "Electron Density": "Electron_Density",
    "Charge Density": "Charge_Density",
}

# Step 2: 上下文评分关键词 (用于寻找分子定义的段落)
DEFINITION_KEYWORDS = [
    "synthesis", "prepared", "preparation", "scheme", "figure", "structure", 
    "obtained", "reaction", "synthesized", "compound", "derivative", "r ="
]
