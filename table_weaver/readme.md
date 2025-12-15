## 使用方法

### 1. 运行完整评估

```bash
cd /home/lilin/table_weaver
python run_evaluation.py
```

### 2. 使用自定义参数运行

```bash
python finqa_evaluator.py --config ../agent_config/example.yaml --data ../test_data/finqa.json --output ./results --max-samples 100
```

### 3. 分析结果

```bash
python analyze_results.py --results ./results/final_results.json --output ./results
```

## 评估流程

1. 加载 FINQA 数据集
2. 对每个样本执行以下步骤：
   - 加载表格数据
   - 使用 TableZoomer 生成表格描述
   - 执行查询并获取预测结果
   - 比较预测结果与正确答案
3. 计算准确率和其他评估指标
4. 保存详细结果和指标

## 结果说明

评估完成后，将在 `results` 目录下生成以下文件：

- `final_results.json`: 所有样本的详细评估结果
- `metrics.json`: 评估指标汇总
- `table_schemas/`: 生成的表格结构描述文件（仅包含表结构相关内容）
- `paragraph_schemas/`: 生成的段落结构化描述文件（仅包含文本+问题相关内容）
- `analysis_report.json`: 详细分析报告
- `accuracy_pie.png`: 准确率可视化图表
- `error_types.png`: 错误类型分布图表

## 注意事项

1. 确保已经正确配置了 TableZoomer 的环境和依赖
2. 确保 `test_data/FINQA/csv` 目录下存在所有需要的 CSV 文件
3. 评估过程可能需要较长时间，具体取决于样本数量和计算资源

# 添加缺失的函数 - longest_common_subsequence
```python
# 实现：返回两个字符串的最长公共子序列长度（经典动态规划）
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    返回 text1 和 text2 的最长公共子序列 (LCS) 的长度。
    时间复杂度 O(len(text1) * len(text2))，空间使用滚动数组优化至 O(min(n,m)).
    """
    if not text1 or not text2:
        return 0
    # 确保短串为 cols，长串为 rows，节省空间
    if len(text1) < len(text2):
        short, long = text1, text2
    else:
        short, long = text2, text1
    cols = len(short)
    prev = [0] * (cols + 1)
    for ch in long:
        curr = [0] * (cols + 1)
        for j in range(cols):
            if ch == short[j]:
                curr[j+1] = prev[j] + 1
            else:
                curr[j+1] = max(prev[j+1], curr[j])
        prev = curr
    return prev[cols]
```

            # 使用Weaver的ask方法处理问题，这会自动包含计划生成和验证
            if self.weaver:
                logger.info("使用Weaver的完整流程处理问题...")
                
                # 加载表格数据
                table = self._load_table_data()
                if table is None:
                    logger.error("无法加载表格数据")
                    return {
                        "answer": "Error: Failed to load table data",
                        "success": False,
                        "execute_state": "fail"
                    }
                
                # 创建临时列描述文件
                import tempfile
                import json
                import os
                
                column_descriptions = {col: f"Column {col}" for col in table.columns}
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(column_descriptions, f)
                    temp_desc_file = f.name
                
                try:
                    # 构建问题对象，使用column_description_file参数
                    question_obj = {
                        'question': query,
                        'table_name': self.table_desc.get('name', 'table') if isinstance(self.table_desc, dict) else 'table',
                        'table': table,
                        'paragraphs': paragraph_schema,
                        # 使用column_description_file参数，Weaver会读取这个文件而不生成新描述
                        'column_description_file': temp_desc_file
                    }
                    
                    # 调用Weaver的ask方法
                    result = self.weaver.ask(question_obj)
                    logger.info("Weaver处理完成")
                    
                    # 提取答案
                    final_answer = result.answer
                    logger.info(f"提取的答案: {final_answer}")
                    
                    return {
                        "answer": final_answer,
                        "success": True,
                        "execute_state": "success"
                    }
                finally:
                    # 删除临时文件
                    if os.path.exists(temp_desc_file):
                        os.remove(temp_desc_file)