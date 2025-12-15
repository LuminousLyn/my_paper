#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析FINQA评估结果
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(results_path):
    """加载评估结果"""
    with open(results_path, 'r', encoding='utf8') as f:
        return json.load(f)

def analyze_errors(results):
    """分析错误类型"""
    error_types = {}
    for result in results:
        if not result['is_correct']:
            error = result.get('error', 'Incorrect prediction')
            if error in error_types:
                error_types[error] += 1
            else:
                error_types[error] = 1
    return error_types

def generate_report(results, output_dir):
    """生成分析报告"""
    # 计算基本指标
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    # 分析错误
    error_types = analyze_errors(results)
    
    # 创建报告字典
    report = {
        'total_samples': total,
        'correct_samples': correct,
        'accuracy': accuracy,
        'error_types': error_types,
        'detailed_results': results
    }
    
    # 保存报告
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, 'w', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成可视化
    generate_visualizations(report, output_dir)
    
    print(f"Analysis report saved to {report_path}")
    print(f"Total samples: {total}")
    print(f"Correct samples: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nError types:")
    for error, count in error_types.items():
        print(f"  {error}: {count}")

def generate_visualizations(report, output_dir):
    """生成可视化图表"""
    try:
        # 准确率饼图
        plt.figure(figsize=(8, 6))
        labels = ['Correct', 'Incorrect']
        sizes = [report['correct_samples'], report['total_samples'] - report['correct_samples']]
        colors = ['#4CAF50', '#F44336']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('FINQA Evaluation Accuracy')
        plt.savefig(os.path.join(output_dir, 'accuracy_pie.png'), dpi=300, bbox_inches='tight')
        
        # 错误类型条形图
        if report['error_types']:
            plt.figure(figsize=(10, 6))
            errors = list(report['error_types'].keys())
            counts = list(report['error_types'].values())
            
            plt.barh(errors, counts)
            plt.xlabel('Count')
            plt.title('Error Types Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_types.png'), dpi=300, bbox_inches='tight')
        
        plt.close('all')
        print("Visualizations generated successfully")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze FINQA evaluation results')
    parser.add_argument('--results', type=str, default='./results/final_results.json', 
                        help='Path to the results file')
    parser.add_argument('--output', type=str, default='./results', 
                        help='Directory to save analysis')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载并分析结果
    results = load_results(args.results)
    generate_report(results, args.output)

if __name__ == "__main__":
    main()