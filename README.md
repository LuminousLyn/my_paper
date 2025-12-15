# TableZoomer + Weaver 集成框架（All Projects）

## 📋 项目概述

这是一个整合了 **TableZoomer** 和 **Weaver** 框架的表格问答（Table QA）综合解决方案。该项目实现了从表格数据预处理到自动问答的完整流程，采用 **ReAct 架构**进行动态调整和自适应执行。

**核心能力**：
- 自动化表格数据清洗与标准化
- 智能表格模式识别与生成（Schema Generation）
- 多阶段执行计划生成与验证
- 动态代码生成与执行
- LLM 语义推理与结果验证
- 自适应反馈与错误恢复

---

## 🏗️ 整体架构

```
输入：CSV 表格 + 自然语言查询
  ↓
【第一部分】表格预处理框架
  ├─ 读取 CSV 表格数据
  ├─ 数据清洗与规范化
  ├─ 数据库存储
  ├─ 生成 Table Schema（表结构描述）
  ├─ 生成 Paragraph Schema（段落/文本描述）
  └─ Query-Refine 缩表（行≥10 或 列≥10）
      └─ 生成 Refined Schema
  ↓
【第二部分】Plan-Execute 框架
  ├─ 生成执行计划（基于双 Schema + Query）
  ├─ 验证计划可行性
  ├─ 逐步生成代码/推理语句
  │   ├─ SQL 语句
  │   ├─ Python 代码
  │   └─ LLM 语义推理
  ├─ 逐步执行并返回结果
  └─ 执行状态反馈（Success/Failed）
  ↓
【第三部分】ReAct 集成框架（动态调整）
  ├─ 计划验证失败 → 重新生成计划
  ├─ 代码执行失败 → 重新生成代码
  ├─ 信息缺失 → 重新生成双 Schema
  └─ 缺失字段检测 → LLM 反馈修正
  ↓
【输出】
  ├─ 最终答案
  ├─ 执行计划
  ├─ 执行日志
  └─ 执行状态
```

---

## 📁 项目结构

### all_projects/

```
all_projects/
│
├── TableZoomer/                      # 核心框架实现
│   ├── table_agent1.py              # 主程序入口，整合所有流程
│   ├── logging_config.py            
│   ├
│   │
│   ├── actions/                     # 核心执行模块
│   │   ├── enhanced_executor.py     # Plan-Execute 框架实现
│   │   ├── table_desc.py            # Table Schema 生成
│   │   ├── paragraph_schema.py      # Paragraph Schema 生成
│   │   ├── query_analyse.py         # Query 分析与精化
│   │   ├── program_write.py         # SQL/代码生成
│   │   ├── llm_actions.py           # LLM 调用接口
│   │   ├── summarize.py             # 结果总结
│   │   └── weaver_executor.py       # Weaver 执行器适配层
│   │
│   ├── agent_config/                # 配置文件
│   │   ├── weaver_config.yaml       # Weaver 框架配置
│   │   ├── qwen3-8b_api.yaml        # 通义千问 API 配置
│   │   └── example.yaml             # 配置示例
│   │
│   ├── prompts/                     # 提示词库
│   │   ├── code_generate_prompt_*.txt      # 代码生成提示词
│   │   ├── final_answer_prompt_*.txt       # 最终答案提示词
│   │   ├── query_refine_*.txt              # Query 精化提示词
│   │   ├── react_prompt_*.txt              # ReAct 集成提示词
│   │   ├── table_desc_prompt_*.txt         # 表描述提示词
│   │   ├── paragraph_schema_prompt.txt     # 段落 Schema 提示词
│   │   └── weaver/                  # Weaver 相关提示词
│   │
│   ├── roles/                       
│   │   ├── query_planner.py         # Query 规划器
│   │   ├── code_generator.py        # 代码生成器
│   │   ├── answer_formatter.py      # 答案格式化器
│   │   ├── table_describer.py       # 表描述器
│   │   └── llm_chat.py              # LLM 聊天接口
│   │
│   ├── MetaGPT/                     # MetaGPT 框架（子项目）
│   ├── results/                     # 执行结果输出
│   ├── logs/                        # 日志文件
│   └── datasets/                    # 数据集（可选）
│
├── weaver/                          # 基础框架库
│   ├── weaver/                      # 核心模块
│   │   ├── core/
│   │   │   ├── weaver.py            # 主 TableQA 类
│   │   │   ├── base.py              # 基础类
│   │   │   └── weaver_multi.py      # 多表支持
│   │   ├── data/
│   │   │   ├── loader.py            # 数据加载器
│   │   │   ├── preprocessor.py      # 数据预处理
│   │   │   └── validators.py        # 数据验证
│   │   ├── llm/
│   │   │   └── client.py            # LLM 客户端
│   │   ├── prompts/
│   │   │   ├── loader.py            # 提示词加载器
│   │   │   └── builtin_prompts.py   # 内置提示词
│   │   ├── database/
│   │   │   ├── manager.py           # 数据库管理
│   │   │   └── models.py            # 数据模型
│   │   └── config/
│   │       ├── settings.py          # 配置设置
│   │       └── logging_config.py    # 日志配置
│   │
│   ├── datasets/                    # 数据集集合
│   │   ├── finqa.json              # FinQA 数据集
│   │   ├── tabfact.json            # TabFact 数据集
│   │   ├── wikitq.json             # WikiTableQuestions 数据集
│   │   ├── ott-qa.json             # OTT-QA 数据集
│   │   ├── california_schools.json  # 加州学校数据集
│   │   └
│   │
│   ├── prompts/                     # 提示词库
│   │   ├── common/                  # 通用提示词
│   │   ├── finqa/                   # FinQA 特定提示词
│   │   ├── tabfact/                 # TabFact 特定提示词
│   │   └── wikitq/                  # WikiTableQuestions 特定提示词
│   │
│   ├── results/                     # 评估结果
│   ├── requirements.txt             # 依赖声明
│   ├── setup.py                     # 安装脚本
│   └── README.md                    # Weaver 项目文档
│
└── table_weaver/                    # 整合后的评估流程
    ├── run_evaluation.py            # 评估入口脚本
    ├── finqa_evaluate.py            # FinQA 评估模块
    ├── analysis.py                  # 结果分析
    ├── datasets/                    # 数据集
    ├── prompts/                     # 提示词
    ├── results/                     # 评估结果
    │   ├── final_results.json       # 最终评估结果
    │   ├── metrics.json             # 评估指标
    │   ├── results_*.json           # 分批处理结果
    │   ├── table_schemas/           # 生成的表 Schema
    │   └── paragraph_schemas/       # 生成的段落 Schema
    └── logs/                        # 评估日志
```

---



---

## 🔄 核心工作流

### Phase 1: 表格预处理

```python
from TableZoomer.table_agent1 import TableZoomer

# 初始化
zoomer = TableZoomer(config_file='agent_config/weaver_config.yaml')

# 生成 Table Schema（列信息、数据类型、示例值）
table_schema = zoomer.get_table_schema(
    table_file='data.csv',
    save_path='schema.json',
    paragraphs=None  # 可选：表格相关的段落文本
)

# 生成 Paragraph Schema（如果有相关文本）
paragraph_schema = zoomer.generate_paragraph_schema(
    paragraphs=['revenue increased by 10%', ...],
    table_schema=table_schema,
    question="Which company has the highest revenue?"
)
```

### Phase 2: Plan-Execute 执行

```python
# 调用 Plan-Execute 框架
answer, log = zoomer.execute_qa(
    query="Which company has the highest revenue?",
    table_file='data.csv',
    table_schema_path='schema.json',
    paragraphs=None
)

# 返回内容
# {
#     "answer": "Final Answer",
#     "success": True,
#     "execute_state": "success",
#     "plan": "Step 1: ...",
#     "steps": [...],
#     "execution_results": [...]
# }
```

### Phase 3: 结果处理

```python
from TableZoomer.postprocess import ResultProcessor

processor = ResultProcessor()
formatted_answer = processor.format_answer(answer)
```

---

## 📊 支持的数据集

### 数据集列表

| 数据集 | 位置 | 表格数 | 问题数 | 特点 |
|-------|------|--------|--------|------|
| **FinQA** | `weaver/datasets/FINQA/` | 3000+ | 8000+ | 财务报表问答 |
| **TabFact** | `weaver/datasets/TabFact/` | 150K+ | 175K+ | 表格事实验证 |
| **WikiTableQuestions** | `weaver/datasets/WikiTableQuestions/` | 22K | 22K | 维基百科表格问答 |
| **OTT-QA** | `weaver/datasets/OTT-QA/` | - | 10K+ | 开放表格问答 |
| **California Schools** | `weaver/datasets/california_schools/` | - | - | 学校数据问答 |

### 数据集格式

```json
{
  "table_id": "finqa_0001",
  "table": [
    ["Company", "Revenue (M)", "Year"],
    ["Apple", "365817", "2021"],
    ["Microsoft", "198252", "2021"]
  ],
  "question": "Which company has higher revenue?",
  "answer": "Apple",
  "supporting_facts": [...],
  "paragraphs": ["Apple's revenue increased..."]
}
```

---

## 🔧 主要模块说明

### 1. `enhanced_executor.py` - Plan-Execute 框架核心

**职责**：生成执行计划、验证计划、逐步执行

**关键方法**：
- `generate_plan()` - 基于双 Schema 生成计划
- `verify_plan()` - 验证计划的可行性
- `execute_step()` - 执行单个步骤
- `process_question()` - 端到端的问题处理

**特性**：
- ✅ 传递 `paragraph_schema` 到完整的执行流程
- ✅ 逐步代码生成与执行
- ✅ 执行结果反馈与状态跟踪

### 2. `table_desc.py` - Table Schema 生成

**职责**：分析表格结构，生成表描述

**功能**：
- 列信息提取（名称、类型、示例值）
- 数据类型推断
- 数据质量评估
- Schema 导出（JSON 格式）

### 3. `paragraph_schema.py` - Paragraph Schema 生成

**职责**：处理表格相关的文本描述

**功能**：
- 段落结构化处理
- 关键信息提取
- Schema 映射到表列
- 文本-表格关联

### 4. `query_analyse.py` - Query 分析与精化

**职责**：分析查询意图，进行动态精化

**流程**：
1. Query 意图识别
2. Query 类型分类（计算、比较、筛选等）
3. 必要列识别
4. 必要行筛选（Query-Refine）



### 6. `roles/` - MetaGPT 角色

**角色分工**：
- `QueryPlanner` - 规划执行计划
- `CodeGenerator` - 生成执行代码
- `AnswerFormatter` - 格式化最终答案
- `TableDescriber` - 生成表描述

---

## 🎯 ReAct 集成机制

### 错误恢复策略

```
计划验证失败
  └─ 理由分析
     └─ 重新生成计划
        └─ 继续执行

代码执行失败（如：信息缺失、字段不存在）
  └─ 错误分析
     └─ LLM 反馈缺失信息
        └─ 重新生成双 Schema
           └─ 重新生成代码
              └─ 重新执行

执行状态为 Failed
  └─ 分析失败原因
     └─ 重新生成该步骤代码
        └─ 重新执行
```

### 日志跟踪

所有执行过程都有详细日志记录：

```
logs/
  ├── table_agent1.log          # 主程序日志
  ├── code_generation.log       # 代码生成日志
  ├── enhanced_executor.log     # 执行器日志
  └── plan_verification.log     # 计划验证日志
```

---

## 💾 数据库管理

### 表格存储

```python
# 自动存储
from weaver.database.manager import DatabaseManager

db_manager = DatabaseManager(db_type='sqlite', path='./data/tables.db')

# 表格会自动存储为表
# table_name: <file_name_cleaned>
# columns: 按原表结构
```

### 查询接口

```python
# 从数据库查询
result = db_manager.query(
    table_name='company_financial_2021',
    columns=['Company', 'Revenue'],
    where_conditions={'Year': 2021}
)
```

---

## 📈 评估与分析

### 运行评估

```bash
cd table_weaver

# 对 FinQA 数据集评估
python run_evaluation.py \
    --config ../TableZoomer/agent_config/weaver_config.yaml \
    --dataset finqa \
    --samples 100  # 评估 100 个样本

# 对 TabFact 数据集评估
python run_evaluation.py \
    --config ../TableZoomer/agent_config/weaver_config.yaml \
    --dataset tabfact \
    --samples 100
```

### 查看结果

```bash
# 最终评估结果
cat results/final_results.json

# 评估指标
cat results/metrics.json

# 单个样本结果（分批存储）
cat results/results_100.json
```

### 结果分析

```python
# 使用分析脚本
python analysis.py

# 输出内容
# - 准确率 (EM, F1)
# - 错误分类统计
# - 性能分布
# - 失败案例分析
```

---

## 🛠️ 开发指南

### 添加新的数据集

1. 在 `weaver/datasets/` 创建新文件夹
2. 将数据集放入该文件夹
3. 在 `table_weaver/run_evaluation.py` 中注册数据集
4. 运行评估脚本

### 自定义 LLM 提示词

1. 编辑 `TableZoomer/prompts/` 中的提示词文件
2. 支持模板变量：`{query}`, `{table_schema}`, `{paragraph_schema}` 等
3. 在代码中通过 `load_prompt()` 加载

### 扩展执行器

修改 `TableZoomer/actions/enhanced_executor.py`：

```python
class WeaverBasedCodeExecutor:
    def execute_step(self, step, df, query, previous_results):
        """
        自定义执行逻辑
        
        Args:
            step: {'step': 1, 'tool': 'sql', 'code': '...'}
            df: DataFrame
            query: 原始查询
            previous_results: 上一步结果
        
        Returns:
            {'success': bool, 'result': any, 'error': str}
        """
        # 你的实现
        pass
```

---

## 🐛 常见问题

### Q1: 如何处理超大表格？

**A**: 使用 Query-Refine 机制：

```python
# 自动启用
table_schema = zoomer.get_table_schema(
    table_file='large_table.csv',
    # 行 ≥ 10 或 列 ≥ 10 时自动进行缩表
)

# 查看缩表结果
print(table_schema['refined_schema'])
```

### Q2: 如何调试执行计划？

**A**: 查看详细日志：

```bash
tail -f logs/table_agent1.log
# 或
tail -f logs/code_generation.log
```

### Q3: API 调用失败怎么办？

**A**: 检查配置文件和 API 密钥：

```bash
# 验证配置
python -c "from weaver.config.settings import WeaverConfig; print(WeaverConfig.load())"

# 测试 LLM 连接
python -c "from weaver.llm.client import LLMClient; client = LLMClient(); print(client.chat('hello'))"
```

---

## 📚 相关文档

- [Weaver 框架文档](weaver/README.md)
- [TableZoomer 框架文档](TableZoomer/README.md)
- [table_weaver 评估指南](table_weaver/readme.md)

---


