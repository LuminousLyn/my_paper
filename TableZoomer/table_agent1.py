"""
Created Date: 2025-05-22
Author: xiongsishi@chinatelecom.cn

TableZoomer与Weaver框架融合实现
根据FINQA数据集的表格生成表描述→根据查询类型进行缩表生成schema→进入weaver的plan-execute框架
"""
# 在文件开头添加
from logging_config import setup_logging

# 在类初始化之前设置日志
main_logger, code_gen_logger = setup_logging()
logger = main_logger
import os
import sys
import pandas as pd
from pathlib import Path
import json
import yaml
import argparse
import copy
from typing import List, Dict, Tuple, Union, Any
import asyncio
import ast
import time
from tqdm import tqdm
import logging

# 设置日志
import os
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 创建logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建主日志文件的handler，设置延迟写入为False，确保实时写入
file_handler = logging.FileHandler(os.path.join(log_dir, 'table_agent1.log'), delay=False)
file_handler.setFormatter(formatter)

# 创建代码生成专用日志文件的handler
code_gen_handler = logging.FileHandler(os.path.join(log_dir, 'code_generation.log'), delay=False)
code_gen_handler.setFormatter(formatter)

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

# 创建代码生成专用的logger
code_gen_logger = logging.getLogger('code_generation')
code_gen_logger.setLevel(logging.INFO)
code_gen_logger.handlers = []
code_gen_logger.addHandler(code_gen_handler)
code_gen_logger.addHandler(stream_handler)
code_gen_logger.propagate = False

# 获取当前文件所在的目录
CUR_ROOT = Path(__file__).parent.resolve()

# 将MetaGPT项目添加到Python路径
sys.path.insert(0, os.path.join(CUR_ROOT, 'MetaGPT'))
# 添加Weaver项目到Python路径
sys.path.insert(0, '/home/lilin/weaver')

# 导入必要的模块
from metagpt.logs import logger as metagpt_logger
from metagpt.config2 import Config
from roles import QueryPlanner, AnswerFormatter, LLMChat, TableDescriber
from actions.table_desc import get_refined_table_schema
from actions.paragraph_schema import ParagraphSchema
from actions.enhanced_executor import WeaverBasedCodeExecutor

# 导入Weaver相关模块
from weaver.core.weaver import TableQA
from weaver.config.settings import WeaverConfig


class TableZoomer():
    def __init__(self,
            config_file,
            max_react_round=5,
            ):
        # 初始化LLM和提示词配置
        self._init_llm_prompt_config(config_file)
        # 初始化角色
        self._init_roles()
        self.max_react_round = max_react_round
        # 初始化Weaver框架
        self._init_weaver()
        # 初始化Weaver的plan-execute执行器
        self._init_weaver_executor()

    def _init_llm_prompt_config(self, config_file):
        """初始化LLM配置和提示词模板"""
        # 读取YAML配置文件
        with open(config_file, 'r') as file:
            yaml_config = yaml.safe_load(file)
        
        # 加载提示词模板
        self.react_prompt = open(os.path.join(CUR_ROOT, yaml_config['prompt_template']['react']), 'r', encoding='utf8').read()
        self.table_desc_prompt = open(os.path.join(CUR_ROOT, yaml_config['prompt_template']['table_desc']), 'r', encoding='utf8').read()
        self.query_expansion_prompt = open(os.path.join(CUR_ROOT, yaml_config['prompt_template']['query_expansion']), 'r', encoding='utf8').read()
        self.paragraph_schema_prompt = open(os.path.join(CUR_ROOT, 'prompts', 'paragraph_schema_prompt.txt'), 'r', encoding='utf8').read()
        self.answer_summary_prompt = open(os.path.join(CUR_ROOT, yaml_config['prompt_template']['answer_summary']), 'r', encoding='utf8').read()
        
        # 加载LLM配置
        self.react_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['react'])))
        self.table_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['table_desc'])))
        self.query_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['query_expansion'])))
        self.summary_llm_config = Config.from_yaml_file(Path(os.path.join(CUR_ROOT, 'agent_config', yaml_config['llm_config']['answer_summary'])))

    def _init_roles(self):
        """初始化所有需要的角色"""
        self.query_planner_role = QueryPlanner(llm_config=self.query_llm_config, prompt_template=self.query_expansion_prompt)
        self.answer_formatter_role = AnswerFormatter(llm_config=self.summary_llm_config, prompt_template=self.answer_summary_prompt)
        self.llm = LLMChat(llm_config=self.react_llm_config, prompt_template=self.react_prompt)
    
    def _init_weaver(self):
        """初始化Weaver框架"""
        logger.info("初始化Weaver框架...")
        
        # 从配置文件创建Weaver配置
        config_path = os.path.join(CUR_ROOT, 'agent_config', 'weaver_config.yaml')
        weaver_config = WeaverConfig.from_env()
        
        # 如果配置文件存在，则加载其中的设置
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # 应用数据库配置
            if 'database' in config_data:
                db_config = config_data['database']
                if 'type' in db_config:
                    weaver_config.database.db_type = db_config['type']
                if 'path' in db_config:
                    weaver_config.database.db_path = db_config['path']
# 应用LLM配置
            if 'llm' in config_data:
                llm_config = config_data['llm']
                if 'model' in llm_config:
                    # 构建完整的模型名称，添加provider前缀
                    if 'api_type' in llm_config:
                        weaver_config.llm.model = f"{llm_config['api_type']}/{llm_config['model']}"
                    else:
                        # 如果没有指定api_type，默认使用openai
                        weaver_config.llm.model = f"openai/{llm_config['model']}"
                if 'temperature' in llm_config:
                    weaver_config.llm.temperature = llm_config['temperature']
                if 'max_tokens' in llm_config:
                    weaver_config.llm.max_tokens = llm_config['max_tokens']
                if 'timeout' in llm_config:
                    weaver_config.llm.timeout = llm_config['timeout']
                if 'base_url' in llm_config:
                    weaver_config.llm.api_base = llm_config['base_url']  # 注意这里使用api_base而不是base_url
                if 'api_key' in llm_config:
                    # 设置API密钥到环境变量
                    os.environ['OPENAI_API_KEY'] = llm_config['api_key']
            
            # 应用提示词目录配置
            if 'prompts' in config_data and 'dir' in config_data['prompts']:
                weaver_config.prompts_dir = Path(CUR_ROOT) / config_data['prompts']['dir']
        
        # 初始化Weaver实例
        self.weaver_qa = TableQA(config=weaver_config)
        logger.info("Weaver框架初始化完成")

    def _init_weaver_executor(self):
        """初始化Weaver的plan-execute执行器"""
        # 使用已经在_init_weaver中处理好的weaver_config对象
        # 修复Weaver的导入问题
        self.weaver_executor = WeaverBasedCodeExecutor(config=self.weaver_qa.config)
        logger.info("Weaver执行器初始化完成")

    def act_pipeline(self, query, table_schema):
        """TableZoomer的动作管道，处理查询并优化表格schema"""
        logger.info('** Step1: Schema Refining between table and query.')

        # Step 1.1: 查询规划
        logger.info('1.1 Query Planning...')
        try:
            msg = json.dumps({"query": query, "table_desc": table_schema}, ensure_ascii=False)
            result = asyncio.run(self.query_planner_role.run(msg))
            # 兼容 Message 或 str 返回
            if hasattr(result, "content"):
                query_rsp = json.loads(result.content)
            else:
                query_rsp = json.loads(result)
        except Exception as e:
            logger.info(f'Exception: {e}')
            logger.info('Exceed max input length! Trimming...')
            if 'description' in table_schema and len(table_schema['description']) > 2000:
                table_schema['description'] = table_schema['description'][:2000]
            elif 'cell_example' in table_schema:
                del table_schema['cell_example']
            msg = json.dumps({"query": query, "table_desc": table_schema}, ensure_ascii=False)
            result = asyncio.run(self.query_planner_role.run(msg))
            if hasattr(result, "content"):
                query_rsp = json.loads(result.content)
            else:
                query_rsp = json.loads(result)
        query_analysis = query_rsp['rsp']

        try:
            query_expansions = json.loads(query_analysis)
        except:
            query_expansions = ast.literal_eval(query_analysis)

        # Step 1.2: 表格schema优化
        logger.info('1.2 Table schema refinement...')
        relevant_column_list = []
        type = query_expansions[0]['type']
        row_match_list = query_expansions[0]['row_match_list']

        relevant_column_list.extend(
            [r for q in query_expansions for r in q['relevant_column_list'] if r in table_schema['column_list']])
        relevant_column_list = list(set(relevant_column_list))
        
        if len(relevant_column_list) > 0:
            # 生成优化后的表格schema
            refined_table_schema = get_refined_table_schema(table_schema, relevant_column_list, type, row_match_list)

            # 格式化cell_example
            raw_cell_example = refined_table_schema.get("cell_example", [])
            if raw_cell_example != []:
                raw_cell_example = raw_cell_example[0]
            else:
                logger.info("cell example not found")
            
            # 设置格式为结构化格式，便于后续处理
            fmt = "struct"
            try:
                if isinstance(raw_cell_example, list) and len(raw_cell_example) == 1 and isinstance(raw_cell_example[0], list):
                    flat_example = raw_cell_example[0]
                else:
                    flat_example = raw_cell_example

                df = pd.DataFrame(flat_example)

                if fmt == "markdown":
                    refined_table_schema["cell_example"] = df.to_markdown(index=False)
                elif fmt == "struct":
                    refined_table_schema["cell_example"] = {
                        "header": df.columns.tolist(),
                        "rows": df.values.tolist()
                    }
                elif fmt == "str":
                    refined_table_schema["cell_example"] = df.to_string()

            except Exception as e:
                logger.warning(f"Failed to format cell_example ({fmt}): {e}")
                # 格式化失败时保持原样
                refined_table_schema["cell_example"] = raw_cell_example

            logger.info(f'Refined table schema generated successfully')
        else:
            refined_table_schema = table_schema
            logger.info('Table schema refinement FAILED! Using the complete table schema without refinement.')

        return refined_table_schema, relevant_column_list, query_rsp

    def _init_table_desc(self, table_file, table_schema_path=None, paragraphs=None):
        """初始化表格描述"""
        logger.info(f'0. Generate table description.')
        
        # 创建或读取表格schema
        logger.info("Get table schema...")
        if table_schema_path is not None and os.path.exists(table_schema_path):
            table_desc = json.load(open(table_schema_path, 'r', encoding='utf8'))
            logger.info(f'Read table schema from {table_schema_path}')
            # 如果table_desc中包含paragraph_schema，将其移除
            if "paragraph_schema" in table_desc:
                del table_desc["paragraph_schema"]
        else:
            # 调用get_table_schema生成表格描述
            table_desc = self.get_table_schema(table_file, table_schema_path)
        
        # 保存纯净的table_desc（不包含paragraph_schema）
        if table_schema_path is not None:
            os.makedirs(os.path.dirname(table_schema_path), exist_ok=True)
            with open(table_schema_path, 'w', encoding='utf8') as f:
                json.dump(table_desc, f, ensure_ascii=False, indent=2)
        
        return table_desc

    def generate_paragraph_schema(self, paragraphs, table_schema, question, table_file=None):
        """生成段落schema"""
        logger.info("Generating paragraph schema...")
        logger.info(f"Input paragraphs: {paragraphs}")
        logger.info(f"Input question: {question}")
        
        # 处理paragraphs参数
        if isinstance(paragraphs, list):
            paragraph_text = " ".join(paragraphs)
        else:
            paragraph_text = paragraphs
            
        # 如果paragraphs为空，返回默认schema
        if not paragraph_text:
            logger.warning("Empty paragraphs provided, returning default schema")
            return {
                "text_summary": "No paragraph text available.",
                "table_related_text": [{"background_related": "No paragraph text provided to analyze."}],
                "question_related_text": [{"Question_Concept": "No paragraph text provided to analyze."}] if question else []
            }
        
        # 从Weaver配置创建LLM配置
        weaver_llm_config = self._create_llm_config_from_weaver()
            
        # 创建ParagraphSchema实例并运行
        paragraph_schema_action = ParagraphSchema(PROMPT_TEMPLATE=self.paragraph_schema_prompt, llm_config=weaver_llm_config)
        
        try:
            # 异步运行生成paragraph schema
            paragraph_schema_json = asyncio.run(
                paragraph_schema_action.run(paragraph_text, table_schema, question)
            )
            logger.info(f"Paragraph schema raw response: {paragraph_schema_json}")
            
            # 解析JSON响应
            try:
                paragraph_schema = json.loads(paragraph_schema_json)
                
                # 确保所有字段存在
                if "text_summary" not in paragraph_schema:
                    paragraph_schema["text_summary"] = paragraph_text[:200] + "..." if len(paragraph_text) > 200 else paragraph_text
                    
                # 只有在字段完全缺失时才使用默认值，不要覆盖已有的内容
                if "table_related_text" not in paragraph_schema:
                    paragraph_schema["table_related_text"] = [{"background_related": "No table-related information found in the paragraph."}]
                elif not paragraph_schema["table_related_text"]:
                    # 如果字段存在但为空，让paragraph_schema.py中的增强逻辑处理
                    logger.info("table_related_text is empty, allowing paragraph_schema to handle it")
                    
                if "question_related_text" not in paragraph_schema:
                    if question:
                        paragraph_schema["question_related_text"] = [{"Question_Concept": "No direct information found in the paragraph that answers this question."}]
                    else:
                        paragraph_schema["question_related_text"] = []
                elif not paragraph_schema["question_related_text"] and question:
                    # 如果字段存在但为空，让paragraph_schema.py中的增强逻辑处理
                    logger.info("question_related_text is empty, allowing paragraph_schema to handle it")
                
                # 修复<Q-entity>错误
                if "question_related_text" in paragraph_schema:
                    if isinstance(paragraph_schema["question_related_text"], dict):
                        # 处理错误的格式
                        if "<Q-entity>" in paragraph_schema["question_related_text"]:
                            if not question:
                                paragraph_schema["question_related_text"] = []
                            else:
                                value = paragraph_schema["question_related_text"]["<Q-entity>"]
                                paragraph_schema["question_related_text"] = [{question.split()[0] if question.split() else "Question": value}]
                    elif isinstance(paragraph_schema["question_related_text"], list):
                        # 检查列表中的每个元素
                        for i, item in enumerate(paragraph_schema["question_related_text"]):
                            if isinstance(item, dict):
                                for key in list(item.keys()):
                                    if key == "<Q-entity>":
                                        if not question:
                                            paragraph_schema["question_related_text"].pop(i)
                                        else:
                                            value = item[key]
                                            paragraph_schema["question_related_text"][i] = {question.split()[0] if question.split() else "Question": value}
                
                logger.info(f"Final paragraph schema: {paragraph_schema}")
                return paragraph_schema
                
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse paragraph schema JSON: {e}")
                # 返回包含段落内容的默认schema
                return {
                    "text_summary": paragraph_text[:200] + "..." if len(paragraph_text) > 200 else paragraph_text,
                    "table_related_text": [{"background_related": "No table-related information found in the paragraph."}],
                    "question_related_text": [{"Question_Concept": "No direct information found in the paragraph that answers this question."}] if question else []
                }
                
        except Exception as e:
            logger.error(f"Error generating paragraph schema: {e}", exc_info=True)
            # 发生异常时，返回包含段落内容的默认schema
            return {
                "text_summary": paragraph_text[:200] + "..." if len(paragraph_text) > 200 else paragraph_text,
                "table_related_text": [{"background_related": "No table-related information found in the paragraph."}],
                "question_related_text": [{"Question_Concept": "No direct information found in the paragraph that answers this question."}] if question else []
            }

    def _create_llm_config_from_weaver(self):
        """从Weaver配置创建LLM配置"""
        # 使用Weaver配置中的LLM设置
        weaver_llm_config = self.weaver_qa.config.llm
        
        # 创建Config对象
        from metagpt.config2 import Config
        
        # 创建完整的配置结构，包含llm字段
        full_config_dict = {
            "llm": {
                "api_type": "openai",  # 强制使用openai类型，确保兼容性
                "model": weaver_llm_config.model,
                "temperature": weaver_llm_config.temperature,
                "max_tokens": weaver_llm_config.max_tokens,
                "timeout": weaver_llm_config.timeout,
                "api_base": weaver_llm_config.api_base,
                "api_key": os.environ.get('OPENAI_API_KEY', '')  # 从环境变量获取API密钥
            }
        }
        
        # 创建临时配置文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(full_config_dict, f)
            temp_config_path = f.name
        
        # 从临时文件创建Config对象
        llm_config = Config.from_yaml_file(Path(temp_config_path))
        
        # 删除临时文件
        os.unlink(temp_config_path)
        
        logger.info(f"Created LLM config from Weaver: {full_config_dict}")
        return llm_config

    def get_table_schema(self, table_file, save_path=None, cell_example_format='raw', paragraphs=None):
        """
        获取表格schema，基于des.py实现
        
        Args:
            table_file: 表格文件路径
            save_path: 保存路径
            cell_example_format: 单元格示例格式
            paragraphs: 段落文本（可选）
            
        Returns:
            表格schema
        """
        logger.info('** Step0: Table Schema Generation.')
        
        # 创建TableDescriber角色并生成表格描述
        role = TableDescriber(llm_config=self.table_llm_config, prompt_template=self.table_desc_prompt)
        
        # 调用角色生成表格描述
        table_desc = asyncio.run(role.run(
            json.dumps({"table_file": table_file, "desc_save_path": save_path if save_path is not None else ''},
                       ensure_ascii=False)))
        
        logger.info(f"table_desc:\n {table_desc}")
        # 解析返回的表格描述
        table_desc = json.loads(table_desc.content)
        
        # 如果提供了paragraphs，添加到table_desc中
        if paragraphs is not None:
            table_desc["paragraphs"] = paragraphs
            # 如果指定了保存路径，保存更新后的schema
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf8') as f:
                    json.dump(table_desc, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved table schema with paragraphs to {save_path}")
        
        # 根据cell_example_format格式化cell_example
        if cell_example_format != 'raw' and 'cell_example' in table_desc:
            try:
                # 确保我们有一个DataFrame对象来处理
                if isinstance(table_desc['cell_example'], list) and table_desc['cell_example']:
                    # 如果cell_example是列表，尝试转换为DataFrame
                    if isinstance(table_desc['cell_example'][0], dict):
                        # 如果是字典列表，直接创建DataFrame
                        df = pd.DataFrame(table_desc['cell_example'])
                    else:
                        # 如果是其他类型的列表，保持原样
                        df = None
                else:
                    df = None
                
                if df is not None:
                    if cell_example_format == 'str':
                        table_desc['cell_example'] = df.to_string()
                    elif cell_example_format == 'markdown':
                        table_desc['cell_example'] = df.to_markdown(index=False)
                    elif cell_example_format == 'struct':
                        table_desc['cell_example'] = {
                            "header": df.columns.tolist(),
                            "rows": df.values.tolist()
                        }
            except Exception as e:
                logger.warning(f"Failed to format cell_example ({cell_example_format}): {e}")
                # 格式化失败时保持原样
        
        return table_desc

    def execute_qa(self, query, table_file, table_schema_path, paragraphs=None):
        """执行问答流程，TableZoomer负责表处理，Weaver负责plan-execute和代码执行"""
        start_time = time.time()
        logger.info(f'Query: {query}')
        logger.info(f'Table: {table_file}')

        # 预先加载表格数据并进行清洗（使用Weaver的预处理逻辑）
        logger.info("=== 预先加载和清洗表格数据 ===")
        try:
            # 直接读取文件
            if table_file.endswith('.csv'):
                df = pd.read_csv(table_file, encoding='utf8', on_bad_lines='skip')
            elif table_file.endswith('.xlsx'):
                df = pd.read_excel(table_file)
            elif table_file.endswith('.json'):
                df = pd.read_json(table_file)
            else:
                logger.warning(f"Unsupported file format: {table_file}")
                df = None
            
            if df is not None and not df.empty:
                logger.info(f"成功加载表格数据，共 {len(df)} 行，列数: {len(df.columns)}")
                logger.info(f"原始表格列名: {list(df.columns)}")
                
                # 使用Weaver的TablePreprocessor进行表格清洗
                logger.info("使用Weaver的表格预处理逻辑清洗表格数据...")
                from weaver.data.preprocessor import TablePreprocessor
                preprocessor = TablePreprocessor()
                table_name = os.path.basename(table_file).split('.')[0]
                clean_table_name, clean_df = preprocessor.clean_table(df, table_name)
                logger.info(f"表格清洗完成，新表名: {clean_table_name}")
                logger.info(f"清洗后表格列名: {list(clean_df.columns)}")
                logger.info(f"清洗后表格大小: {clean_df.shape}")
                

                
                # 将clean_table_name存储到DataFrame的attrs属性中，供后续使用
                if not hasattr(clean_df, 'attrs'):
                    clean_df.attrs = {}
                clean_df.attrs['clean_table_name'] = clean_table_name
                logger.info(f"✅ 已将clean_table_name存储到DataFrame.attrs: {clean_table_name}")
                
                # 将清洗后的表格保存为临时文件，用于后续生成schema
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    clean_df.to_csv(f, index=False)
                    cleaned_table_file = f.name
                
                logger.info(f"✅ 清洗后的表格已保存到临时文件: {cleaned_table_file}")
            else:
                logger.warning("表格数据为空，使用原始表格文件")
                cleaned_table_file = table_file
        except Exception as e:
            logger.error(f"加载和清洗表格数据失败: {str(e)}")
            logger.info("使用原始表格文件继续处理")
            cleaned_table_file = table_file

        # Step 1: 初始化table_desc（表描述和schema优化）
        table_desc = self._init_table_desc(cleaned_table_file, table_schema_path, paragraphs)

        # 添加clean_table_name到table_desc
        table_desc['clean_table_name'] = clean_table_name
        
        # Step 2: 生成或加载paragraph_schema（如果有paragraphs）
        paragraph_schema = None
        if paragraphs is not None:
            # 从table_schema_path中提取table_id
            table_schema_filename = os.path.basename(table_schema_path)
            table_id = os.path.splitext(table_schema_filename)[0]
            
            # 构建paragraph_schema保存路径
            paragraph_schemas_dir = os.path.join(os.path.dirname(os.path.dirname(table_schema_path)), 'paragraph_schemas')
            os.makedirs(paragraph_schemas_dir, exist_ok=True)
            
            paragraph_schema_path = os.path.join(paragraph_schemas_dir, f"{table_id}.json")
            
            # 检查paragraph_schema是否已经存在
            if os.path.exists(paragraph_schema_path):
                logger.info(f"Loading existing paragraph schema from {paragraph_schema_path}")
                with open(paragraph_schema_path, 'r', encoding='utf8') as f:
                    paragraph_schema = json.load(f)
            else:
                # 生成新的paragraph_schema
                paragraph_schema = self.generate_paragraph_schema(paragraphs, table_desc, query)
                with open(paragraph_schema_path, 'w', encoding='utf8') as f:
                    json.dump(paragraph_schema, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved paragraph schema to {paragraph_schema_path}")
        
        # Step 3: TableZoomer缩表
        refined_table_schema, relevant_column_list, query_rsp = self.act_pipeline(query, table_desc)
        
        # Step 4: 交给weaver plan-execute框架（代码生成与执行）
        logger.info('Starting Weaver plan-execute framework...')
        try:
            result = self.weaver_executor.process_question(query, refined_table_schema, table_desc=table_desc, paragraph_schema=paragraph_schema)
            final_answer = result.get('answer', '').strip()
        except Exception as e:
            logger.error(f"Weaver plan-execute error: {e}")
            final_answer = "Error in plan-execute"

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        logger.info(f"*** Final Answer (elapsed time: {elapsed_time} s) ***")
        logger.info(final_answer)

        log_item = {
            'question': query,
            'response': final_answer,
            'elapsed_time/s': elapsed_time,
            'relevant_column_list': relevant_column_list,
            'query_analysis_response': query_rsp['rsp']
        }
        logger.info("*"*10)
        return final_answer, log_item

    def _execute_original_qa(self, query, table_desc, refined_table_schema, relevant_column_list, query_rsp):
        """原始TableZoomer执行逻辑，作为Weaver失败时的回退"""
        log_item = {
            "question": query,
            "query_analysis_prompt": [query_rsp['prompt']],
            "query_analysis_response": [query_rsp['rsp']],
            "relevant_column_list": [relevant_column_list],
        }
        
        # 简化的原始执行逻辑
        thought_process = f"Table zooming completed. Relevant columns: {relevant_column_list}"
        msg = json.dumps({"query": query, "thought_process": thought_process, "table_schema": refined_table_schema}, ensure_ascii=False)
        
        try:
            # 直接格式化答案
            response = json.loads(asyncio.run(self.answer_formatter_role.run(msg)).content)
            return response['response']
        except Exception as e:
            logger.error(f"Error in fallback execution: {e}")
            return f"Error processing query: {str(e)}"

    def simple_voting(self, query, table_file, table_schema_path, k=5, paragraphs=None):
        """简单投票机制，多次执行取多数结果"""
        responses = []
        log_items = []
        logger.info(f"Simple Voting! \nQuery: {query}")

        for i in range(k):
            logger.info(f'{i} infer...')
            try:
                response, log_item = self.execute_qa(query, table_file, table_schema_path, paragraphs)
            except Exception as e:
                logger.error(f'Simple voting error: {e}')
                response = "fail"
                log_item = None

            responses.append(response)
            log_items.append(log_item)

        # 过滤失败的响应
        filtered_responses = []
        filtered_log_items = []
        for response, log_item in zip(responses, log_items):
            if response != "fail":
                filtered_responses.append(response)
                filtered_log_items.append(log_item)

        if not filtered_responses:
            return "fail", None

        # 统计响应频率
        frequency = {}
        for response in filtered_responses:
            if response in frequency:
                frequency[response] += 1
            else:
                frequency[response] = 1

        # 选择最频繁的响应
        vote_result = max(frequency, key=frequency.get)
        index = filtered_responses.index(vote_result)
        vote_log_item = filtered_log_items[index]
        vote_log_item['vote_answer_list'] = filtered_responses
        vote_log_item['vote_relevant_column_list'] = [i['relevant_column_list'] for i in filtered_log_items]
        
        logger.info(f'Simple Voting: \nQuery: {query}\nAnswers: {filtered_responses}\nFinal Answer: {vote_result}')

        return vote_result, vote_log_item


# 主函数示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TableZoomer with Weaver Integration')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--query', type=str, required=True, help='User query')
    parser.add_argument('--table', type=str, required=True, help='Path to table file')
    parser.add_argument('--schema', type=str, help='Path to save/generate table schema')
    parser.add_argument('--paragraphs', type=str, help='Path to paragraphs file (optional)')
    
    args = parser.parse_args()
    
    # 初始化TableZoomer
    zoomer = TableZoomer(config_file=args.config)
    
    # 加载段落（如果提供）
    paragraphs = None
    if args.paragraphs and os.path.exists(args.paragraphs):
        with open(args.paragraphs, 'r', encoding='utf8') as f:
            paragraphs = f.read()
    
    # 执行问答
    answer, log = zoomer.execute_qa(
        query=args.query,
        table_file=args.table,
        table_schema_path=args.schema or os.path.splitext(args.table)[0] + '_schema.json',
        paragraphs=paragraphs
    )
    
    print(f"\nFinal Answer: {answer}")
"""
@Desc: column linking.
@Author: xiongsishi
@Date: 2025-05-23.
"""

import json
from metagpt.actions import Action
import re
import ast

# 实现extract_from_content函数
def extract_from_content(content):
    """从内容中提取JSON格式的响应"""
    # 尝试从内容中提取JSON
    if content.startswith("```json"):
        content = content.replace("```json", "").strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    
    # 尝试查找并解析JSON对象
    import json
    try:
        # 首先尝试直接解析整个内容
        json.loads(content)
        return content
    except json.JSONDecodeError:
        # 如果失败，尝试提取JSON对象
        if "{" in content:
            # 使用JSONDecoder的raw_decode方法来提取有效的JSON
            decoder = json.JSONDecoder()
            start = content.find("{")
            try:
                obj, idx = decoder.raw_decode(content[start:])
                return content[start:start+idx]
            except json.JSONDecodeError:
                # 如果raw_decode也失败，尝试找到最外层的{}对
                brace_count = 0
                start_idx = content.find("{")
                if start_idx == -1:
                    return content
                
                for i in range(start_idx, len(content)):
                    if content[i] == "{":
                        brace_count += 1
                    elif content[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return content[start_idx:i+1]
    
    # 如果所有尝试都失败，返回原始内容
    return content


def contains_unicode_surrogate(text):

    for char in text:
        if '\uD800' <= char <= '\uDBFF':
            return char, True
    return '', False


class QueryExpansion(Action):
    """ Decompose and expand query questions. """
    name: str = "QueryExpansion"

    async def run(self, inputs: str):
        inputs = json.loads(inputs)
        query, table_desc = inputs['query'], inputs['table_desc']

        column_list = table_desc['column_list']
        table_desc = json.dumps(table_desc, ensure_ascii=False, indent=4)
        prompt = self.PROMPT_TEMPLATE.replace("{query}", query).replace("{table_schema}", table_desc)
        print("prompt:",prompt, "\n")
        original_prompt = prompt

        rsp = await self._aask(prompt)

        print("response:", rsp)
        rsp = rsp.strip()
        if rsp.startswith("```json") and rsp.endswith("```"):
            rsp = rsp.replace('```json', '').strip()
            rsp = rsp.replace('```', '')

        rsp = extract_from_content(rsp)

        return json.dumps({"prompt": original_prompt, "rsp": extract_from_content(rsp)}, ensure_ascii=False)

def map_indices_to_names(schema):
    # 假设 schema['column_list'] 是列名列表，schema['columns'] 是数字索引
    if 'column_list' in schema and isinstance(schema['column_list'], list):
        idx_to_name = {str(i): name for i, name in enumerate(schema['column_list'])}
        # 你的后续处理可以用 idx_to_name 做映射
        # 比如 relevant_column_list = [idx_to_name[str(idx)] for idx in relevant_column_indices]