import pandas as pd
import json
import re
from tqdm import tqdm
from src.config import FILES, PROPERTY_MAP
from src.utils import normalize_smiles, is_blacklisted

def clean_value(prop_name, val):
    """数值清洗与单位转换"""
    try:
        v = float(re.search(r'[-+]?\d*\.?\d+', str(val)).group())
        # 简单的 Hartree -> eV 转换示例
        if prop_name in ['HOMO', 'LUMO'] and abs(v) < 1.0: 
            return v * 27.2114
        return v
    except:
        return None

def run():
    print(">>> STEP 4: Post-processing & Validation...")
    data = []
    
    with open(FILES['llm_output'], 'r') as f:
        for line in tqdm(f):
            try:
                item = json.loads(line)
                if 'extracted_data' not in item: continue
                
                for entry in item['extracted_data']:
                    raw_name = entry.get('original_name', '')
                    if is_blacklisted(raw_name): continue
                    
                    # 1. SMILES 验证与脱盐
                    smi = normalize_smiles(entry.get('smiles'), desalt=True)
                    if not smi: continue
                    
                    # 2. 属性提取
                    for k, v in entry.get('properties', {}).items():
                        # 归一化属性名
                        std_prop = PROPERTY_MAP.get(k, k)
                        # 清洗数值
                        clean_v = clean_value(std_prop, v)
                        
                        if clean_v is not None:
                            data.append({
                                "SMILES": smi,
                                "Name": raw_name,
                                "Target": std_prop,
                                "Value": clean_v,
                                "Source": item.get('source_file')
                            })
            except Exception:
                continue

    df = pd.DataFrame(data)
    # 去重
    df = df.drop_duplicates(subset=['SMILES', 'Target'])
    df.to_csv(FILES['dataset_csv'], index=False)
    print(f"Done! Saved {len(df)} valid records to {FILES['dataset_csv']}")

if __name__ == "__main__":
    run()