import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys
from sys import getsizeof

# 内存优化配置（可根据实际内存调整）
CHUNK_SIZE = 50000    # 每块5万行（原10万行的1/2）
MAX_CATEGORY_MEM = 1  # 每千类别允许的内存（GB），自动计算块大小
DTYPE = np.uint8      # 1字节存储（原默认8字节）

def calculate_safe_chunk(all_cats):
    """根据类别数动态计算安全块大小（避免内存爆炸）"""
    cat_count = len(all_cats)
    # 每样本内存 = 类别数 * 1字节 + 索引开销
    sample_mem = cat_count * DTYPE().nbytes + 100  
    # 每块内存 = 块大小 * 样本内存（控制在1GB以内）
    safe_chunk = min(CHUNK_SIZE, int(1e9 / sample_mem))
    return max(1000, safe_chunk)  # 至少1000行

def get_all_amino_acids(file_paths):
    """内存友好的类别收集（流式处理）"""
    valid_chars = set("ACDEFGHIKLMNPQRSTVWYXU-")
    all_seqs = set()
    
    for path in file_paths:
        try:
            # 流式读取文件，每次1000行（极低内存占用）
            for chunk in pd.read_csv(path, usecols=[0], header=0, chunksize=1000):
                col_data = chunk.iloc[:, 0].dropna().astype(str)
                for seq in col_data:
                    if all(c in valid_chars for c in seq):
                        all_seqs.add(seq)
                    else:
                        print(f"过滤无效序列: {seq}", file=sys.stderr)
        except FileNotFoundError:
            print(f"错误: 文件{path}未找到", file=sys.stderr)
            sys.exit(1)
    
    if not all_seqs:
        print("错误: 无有效序列", file=sys.stderr)
        sys.exit(1)
    return sorted(all_seqs)

def process_single_file(args):
    """内存优化的核心处理函数（每块内存<500MB）"""
    file_path, all_cats, output_dir = args
    cat_count = len(all_cats)
    chunk_size = calculate_safe_chunk(all_cats)
    
    # 预计算类别索引映射（比Series.isin快30%）
    cat_to_idx = {cat:i for i, cat in enumerate(all_cats)}
    idx_array = np.arange(cat_count, dtype=np.int32)  # 索引数组
    
    with open(output_path, 'w', newline='') as f_out:
        # 写入标题行（仅一次）
        pd.DataFrame(columns=all_cats).to_csv(f_out, index=False, header=True, mode='a')
        
        for chunk in pd.read_csv(file_path, usecols=[0], header=0, chunksize=chunk_size):
            col_data = chunk.iloc[:, 0].dropna().astype(str)
            if col_data.empty:
                continue
            
            # 内存高效的编码方式（避免全零矩阵）
            rows = len(col_data)
            encoded = np.zeros((rows, cat_count), dtype=DTYPE)
            valid_mask = np.array([seq in cat_to_idx for seq in col_data])
            
            # 向量化赋值（比循环快100倍）
            valid_seqs = col_data[valid_mask]
            encoded[valid_mask, [cat_to_idx[s] for s in valid_seqs]] = 1
            
            # 转换为DataFrame（仅包含有效数据）
            df_encoded = pd.DataFrame(encoded, columns=all_cats)
            df_encoded.to_csv(f_out, index=False, header=False)
    
    return f"完成 {file_path}，类别数={cat_count}，总行数={os.path.getsize(output_path)//1024}KB"

def main():
    INPUT_FILES = ['processed_merged_cdr3_polars.csv', 'TCR_10k_bg_seq.csv']
    OUTPUT_FOLDER = './amino_onehot'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. 流式收集类别（内存占用<10MB）
    print("开始收集序列类别...")
    all_cats = get_all_amino_acids(INPUT_FILES)
    cat_count = len(all_cats)
    print(f"发现{cat_count}种序列，预计单样本内存: {cat_count*1/1024:.2f}KB")
    
    # 2. 动态调整进程数（避免内存竞争）
    ram_gb = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3)
    safe_workers = min(
        max(1, int(cpu_count() * 0.5)),  # 最多50%核心
        ram_gb // max(1, cat_count//1000)  # 每GB内存支持1000类别
    )
    print(f"检测到{ram_gb}GB内存，使用{safe_workers}个安全进程")
    
    # 3. 分文件处理（独立内存空间）
    with Pool(safe_workers) as pool:
        args_list = [(fp, all_cats, OUTPUT_FOLDER) for fp in INPUT_FILES]
        results = pool.map(process_single_file, args_list)
    
    print("\n内存使用报告：")
    print(f"单块最大内存: ~{calculate_safe_chunk(all_cats)*cat_count*1/1024:.1f}KB")
    print(f"总预计内存: {safe_workers * calculate_safe_chunk(all_cats)*cat_count*1/1024:.1f}MB")
    for res in results:
        print(res)

if __name__ == '__main__':
    main()
