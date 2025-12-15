#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINQA评估运行脚本
提供更简单的接口来运行评估
"""

import os
import sys
import argparse
from pathlib import Path

# 将当前目录添加到Python路径
CUR_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(CUR_DIR))

# 添加TableZoomer目录到Python路径
TABLE_ZOOMER_DIR = Path(__file__).parent.parent / 'TableZoomer'
sys.path.insert(0, str(TABLE_ZOOMER_DIR))

from finqa_evaluate import FINQAEvaluator

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run FINQA evaluation with TableZoomer')
    parser.add_argument('--start-index', type=int, default=0, 
                        help='Starting index for evaluation (default: 0)')
    parser.add_argument('--max-samples', type=int, default=None, 
                        help='Maximum number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # 配置参数
    config_file = os.path.join(Path(__file__).parent.parent, 'TableZoomer', 'agent_config', 'example.yaml')
    # 修改data_path指向weaver项目下的finqa.json
    data_path = os.path.join('/home/lilin/weaver/datasets', 'finqa.json')
    output_dir = os.path.join(CUR_DIR, 'results')
    
    # 初始化并运行评估器
    evaluator = FINQAEvaluator(
        config_file=config_file,
        data_path=data_path,
        output_dir=output_dir,
        max_samples=args.max_samples,
        start_index=args.start_index  # 使用命令行参数
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()