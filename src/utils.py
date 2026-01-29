import re
import json
from rdkit import Chem
from rdkit import RDLogger
from src.config import BLACKLIST_PATTERNS

# 关闭 RDKit 警告
RDLogger.DisableLog('rdApp.*')

def normalize_smiles(smiles, desalt=True):
    """
    标准化 SMILES：校验、规范化、可选脱盐
    """
    if not smiles or len(str(smiles)) < 2:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        # 脱盐处理 (只保留最大片段)
        if desalt:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumAtoms())
        
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        return None

def is_blacklisted(text):
    """检查文本是否在黑名单中"""
    s = str(text).strip().lower()
    if len(s) < 2: return True
    for pat in BLACKLIST_PATTERNS:
        if re.search(pat, s):
            return True
    return False

def clean_header_text(text):
    """清洗表头：去单位、特殊符号"""
    text = str(text)
    # 去除 (eV), [kcal] 等
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    # 替换斜杠
    text = text.replace('/', '_')
    return text.strip().strip('-_:\\.')

def append_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")