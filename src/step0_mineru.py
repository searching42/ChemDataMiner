import os
import shutil
import subprocess
import time
from math import ceil
from tqdm import tqdm
from src.config import PDF_DIR, MINERU_OUTPUT_DIR, TEMP_STAGING_DIR, BATCH_SIZE

def safe_symlink(src, dst):
    """创建软链接，如果失败（如Windows权限问题）则回退到复制"""
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def check_is_completed(pdf_filename):
    """检查是否已经处理过"""
    pdf_stem = os.path.splitext(pdf_filename)[0]
    # Mineru 通常会在输出目录创建一个同名的文件夹
    expected_output_dir = MINERU_OUTPUT_DIR / pdf_stem
    if expected_output_dir.exists() and expected_output_dir.is_dir():
        # 简单检查文件夹非空
        if len(list(expected_output_dir.iterdir())) > 0:
            return True
    return False

def run():
    print(">>> STEP 0: Batch Processing PDFs with Mineru...")
    
    if not PDF_DIR.exists():
        print(f"Error: Source PDF directory not found: {PDF_DIR}")
        return

    # 1. 扫描文件
    # 过滤 ._ 开头的 macOS 垃圾文件
    raw_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf') and not f.startswith('._')]
    raw_files.sort()
    
    total_count = len(raw_files)
    todo_list = []
    
    print("Scanning completion status...")
    for pdf in tqdm(raw_files, desc="Checking existing"):
        if not check_is_completed(pdf):
            todo_list.append(pdf)

    wait_count = len(todo_list)
    completed_count = total_count - wait_count
    
    print(f"Total PDFs: {total_count} | Completed: {completed_count} | Pending: {wait_count}")

    if wait_count == 0:
        print("All files processed. Skipping.")
        return

    # 2. 分批处理
    num_batches = ceil(wait_count / BATCH_SIZE)
    print(f"Starting processing in {num_batches} batches...")
    time.sleep(1)

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, wait_count)
        current_batch_files = todo_list[start_idx:end_idx]
        
        batch_num = i + 1
        print(f"\n--- Batch {batch_num}/{num_batches}: Files {start_idx + 1}-{end_idx} ---")

        # 清理并重建临时目录
        if TEMP_STAGING_DIR.exists():
            shutil.rmtree(TEMP_STAGING_DIR)
        TEMP_STAGING_DIR.mkdir(exist_ok=True)

        # 准备暂存区
        for filename in current_batch_files:
            src = PDF_DIR / filename
            dst = TEMP_STAGING_DIR / filename
            safe_symlink(src, dst)

        # 调用 Mineru
        # 注意：这里假设 mineru 命令在系统 PATH 中可用
        cmd = ["mineru", "-p", str(TEMP_STAGING_DIR), "-o", str(MINERU_OUTPUT_DIR)]

        try:
            subprocess.run(cmd, check=True)
            print(f"Batch {batch_num} success.")
        except subprocess.CalledProcessError:
            print(f"Error in batch {batch_num}. Continuing to next batch...")
        except KeyboardInterrupt:
            print("\nUser interrupted.")
            break

    # 清理临时目录
    if TEMP_STAGING_DIR.exists():
        shutil.rmtree(TEMP_STAGING_DIR)

if __name__ == "__main__":
    run()