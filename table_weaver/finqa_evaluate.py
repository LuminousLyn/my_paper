#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINQA数据集评估脚本
用于在FINQA数据集上测试TableZoomer方法并评估准确率
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import asyncio
from tqdm import tqdm
import logging

# 设置日志
# 设置日志
import os
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 创建logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建file handler，设置延迟写入为False，确保实时写入
file_handler = logging.FileHandler(os.path.join(log_dir, 'finqa_evaluation.log'), delay=False)
file_handler.setFormatter(formatter)

# 创建stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# 移除所有现有的handler
logger.handlers = []

# 添加新的handler
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 设置日志记录器的传播属性为False，避免重复记录
logger.propagate = False


# 将TableZoomer项目添加到Python路径
CUR_DIR = Path(__file__).parent.resolve()
TABLE_ZOOMER_DIR = Path(__file__).parent.parent / 'TableZoomer'
sys.path.insert(0, str(TABLE_ZOOMER_DIR))

# 导入现有的TableZoomer类
from table_agent1 import TableZoomer  # 修改这一行，移除TableZoomer前缀

class FINQAEvaluator:
    def __init__(self, config_file, data_path, output_dir, max_samples=None, start_index=0):
        self.config_file = config_file
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.start_index = start_index  # 添加起始索引参数
        # 修改table_dir指向weaver项目下的CSV文件目录
        self.table_dir = os.path.join('/home/lilin/weaver/datasets/FINQA', 'csv')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'table_schemas'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'paragraph_schemas'), exist_ok=True)  # 新增：创建paragraph_schemas目录
        
        # 初始化TableZoomer实例
        logger.info(f"Initializing TableZoomer with config: {config_file}")
        self.table_zoomer = TableZoomer(config_file, max_react_round=5)

    def load_data(self):
        """加载FINQA数据集"""
        logger.info(f"Loading FINQA data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        # 使用start_index从指定位置开始
        if self.start_index > 0:
            data = data[self.start_index:]
            logger.info(f"Starting from index {self.start_index}")
        
        if self.max_samples:
            data = data[:self.max_samples]
        
        logger.info(f"Loaded {len(data)} samples")
        return data

    def run_evaluation(self):
        """运行完整的评估"""
        data = self.load_data()
        results = []
        
        logger.info(f"Starting evaluation of {len(data)} samples")
        logger.info(f"Starting from original index {self.start_index}")
        
        for i, sample in enumerate(tqdm(data, desc="Evaluating")):
            # 更新索引显示，考虑start_index
            original_index = self.start_index + i
            logger.info(f"Processing sample {original_index+1}/{self.start_index+len(data)}")
            result = self.evaluate_sample(sample)
            results.append(result)
            
            # 每处理10个样本保存一次结果
            if (i + 1) % 10 == 0:
                self.save_results(results, f"results_{original_index+1}.json")
        
        # 保存最终结果
        self.save_results(results, "final_results.json")
        
        # 计算准确率
        self.calculate_metrics(results)

    def normalize_answer(self, answer):
        """标准化答案，用于比较"""
        # 移除多余空格
        answer = ' '.join(answer.strip().split())
        # 处理百分比
        if answer.endswith('%'):
            try:
                num = float(answer[:-1])
                return f"{num:.1f}%"
            except:
                pass
        # 处理数字
        try:
            num = float(answer)
            # 根据数字大小选择合适的精度
            if abs(num) >= 1000:
                return f"{num:.0f}"
            elif abs(num) >= 1:
                return f"{num:.1f}"
            else:
                return f"{num:.4f}"
        except:
            pass
        return answer.lower()
    
    def compare_answers(self, normalized_predicted, normalized_target):
        """放宽的答案比较函数"""
        # 1. 完全相同的情况
        if normalized_predicted == normalized_target:
            return True
        
        # 2. 处理百分号差异的情况
        # 检查是否一个有百分号，一个没有，但数值部分相同
        if ('%' in normalized_predicted) != ('%' in normalized_target):
            # 提取数值部分
            pred_num_str = normalized_predicted.replace('%', '')
            target_num_str = normalized_target.replace('%', '')
            try:
                pred_num = float(pred_num_str)
                target_num = float(target_num_str)
                # 数值部分相差小于1算正确
                if abs(pred_num - target_num) < 1:
                    return True
            except:
                pass
        
        # 3. 数值相差小于1的情况（两个都是数字）
        try:
            # 尝试移除百分号并转换为数字比较
            pred_num_str = normalized_predicted.replace('%', '')
            target_num_str = normalized_target.replace('%', '')
            pred_num = float(pred_num_str)
            target_num = float(target_num_str)
            # 数值相差小于1算正确
            if abs(pred_num - target_num) < 1:
                return True
        except:
            pass
        
        return False
    
    def evaluate_sample(self, sample):
        """评估单个样本"""
        try:
            question = sample['question']
            table_file_name = sample['table_file_name']
            target_value = sample['target_value']
            # 获取paragraphs字段，如果不存在则设为None
            paragraphs = sample.get('paragraphs', None)
            
            # 构建完整的表格文件路径
            table_file = os.path.join(self.table_dir, os.path.basename(table_file_name))
            
            if not os.path.exists(table_file):
                logger.error(f"Table file not found: {table_file}")
                return {
                    'question': question,
                    'table_id': sample['table_id'],
                    'target_value': target_value,
                    'predicted_value': None,
                    'is_correct': False,
                    'error': 'Table file not found',
                    'time': 0
                }
            
            # 检查文件是否为空
            if os.path.getsize(table_file) == 0:
                logger.warning(f"Table file is empty (0 bytes): {table_file}")
                return {
                    'question': question,
                    'table_id': sample['table_id'],
                    'target_value': target_value,
                    'predicted_value': None,
                    'is_correct': False,
                    'error': 'Table file is empty',
                    'time': 0
                }
            
            # 生成表格描述文件路径
            table_id = sample['table_id'].replace('/', '_').replace('.pdf', '')
            table_schema_path = os.path.join(self.output_dir, 'table_schemas', f"{table_id}.json")
            
            # 新增：生成paragraph_schema文件路径
            paragraph_schema_path = os.path.join(self.output_dir, 'paragraph_schemas', f"{table_id}.json")

            # 执行查询，传递paragraphs参数
            start_time = time.time()
            answer, log_item = self.table_zoomer.execute_qa(question, table_file, table_schema_path, paragraphs)
            elapsed_time = round(time.time() - start_time, 2)
            
            # 标准化答案
            normalized_predicted = self.normalize_answer(answer)
            normalized_target = self.normalize_answer(target_value)
            
            # 使用放宽的比较函数判断是否正确
            is_correct = self.compare_answers(normalized_predicted, normalized_target)
            
            logger.info(f"Question: {question}")
            logger.info(f"Target: {normalized_target}, Predicted: {normalized_predicted}, Correct: {is_correct}")
            
            return {
                'question': question,
                'table_id': sample['table_id'],
                'target_value': target_value,
                'normalized_target': normalized_target,
                'predicted_value': answer,
                'normalized_predicted': normalized_predicted,
                'is_correct': is_correct,
                'time': elapsed_time,
                'log_item': log_item,
                'paragraph_schema_path': paragraph_schema_path  # 新增：保存paragraph_schema路径
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {str(e)}")
            return {
                'question': sample.get('question', 'Unknown'),
                'table_id': sample.get('table_id', 'Unknown'),
                'target_value': sample.get('target_value', 'Unknown'),
                'predicted_value': None,
                'is_correct': False,
                'error': str(e),
                'time': 0,
                'paragraph_schema_path': None  # 新增：异常情况下也保存路径
            }
    
    def save_results(self, results, filename):
        """保存评估结果，确保包含正确答案和模型输出答案的完整信息"""
        # 创建一个新的结果列表，确保每个结果都包含完整信息
        enhanced_results = []
        for result in results:
            enhanced_result = {
                'question': result.get('question', ''),
                'table_id': result.get('table_id', ''),
                'target_value': result.get('target_value', ''),  # 正确答案
                'normalized_target': result.get('normalized_target', ''),  # 标准化后的正确答案
                'predicted_value': result.get('predicted_value', ''),  # 模型输出答案
                'normalized_predicted': result.get('normalized_predicted', ''),  # 标准化后的模型输出
                'is_correct': result.get('is_correct', False),
                'time': result.get('time', 0),
                'error': result.get('error', ''),
                'paragraph_schema_path': result.get('paragraph_schema_path', ''),  # 新增：paragraph_schema路径
                # 保留log_item中的所有详细信息
                'execution_details': result.get('log_item', {})
            }
            enhanced_results.append(enhanced_result)
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(enhanced_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}, containing target answers and model outputs")
    
    def calculate_metrics(self, results):
        """计算评估指标"""
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        total_time = sum(r['time'] for r in results)
        avg_time = total_time / total if total > 0 else 0
        
        metrics = {
            'total_samples': total,
            'correct_samples': correct,
            'accuracy': accuracy,
            'total_time': total_time,
            'average_time_per_sample': avg_time
        }
        
        # 保存指标
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation completed!")
        logger.info(f"Total samples: {total}")
        logger.info(f"Correct samples: {correct}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per sample: {avg_time:.2f}s")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate TableZoomer on FINQA dataset')
    parser.add_argument('--config', type=str, default='../agent_config/example.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--data', type=str, default='../test_data/finqa.json', 
                        help='Path to the FINQA data file')
    parser.add_argument('--output', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--max-samples', type=int, default=None, 
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--start-index', type=int, default=0, 
                        help='Starting index for evaluation')
    
    args = parser.parse_args()
    
    evaluator = FINQAEvaluator(
        config_file=args.config,
        data_path=args.data,
        output_dir=args.output,
        max_samples=args.max_samples,
        start_index=args.start_index  # 传递start_index参数
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()