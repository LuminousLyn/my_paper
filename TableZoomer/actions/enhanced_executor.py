import json
import asyncio
import logging
import re
import math
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import ast
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any, List

# 导入日志配置
sys.path.append(str(Path(__file__).parent.parent))
from logging_config import setup_logging, log_code_generation

# 设置日志
main_logger, code_gen_logger = setup_logging()
logger = logging.getLogger(__name__)

# 导入Weaver的表格预处理工具
weaver_path = Path("/home/lilin/weaver")
if weaver_path.exists() and str(weaver_path) not in sys.path:
    sys.path.insert(0, str(weaver_path))

from weaver.data.preprocessor import TablePreprocessor


class TableZoomerCodeExecutor:
    """
    TableZoomer自定义代码执行器
    基于TableZoomer.prompts.weaver提示词实现计划生成和验证
    不再复用weaver的完整流程，而是基于自己的思路实现
    """
    def __init__(self, config=None, table_desc=None):
        self.config = config
        self.table_desc = table_desc
        
        # 初始化LLM客户端（如果需要，可以配置自己的LLM）
        self.llm_client = None
        self._init_llm_client()
        
        # 初始化数据库管理器
        self.database = None
        self._init_database()
        
        # 初始化提示词加载器
        self.prompts_dir = Path("/home/lilin/TableZoomer/prompts/weaver")
        self.planner_prompt = self._load_prompt("planner_prompt.txt")
        self.verify_prompt = self._load_prompt("verify_prompt.txt")
        self.execute_prompt = self._load_prompt("execute_prompt.txt")
        self.extract_answer_prompt = self._load_prompt("extract_answer_prompt.txt")
        
        logger.info("✅ 成功初始化TableZoomer代码执行器")
        code_gen_logger.info("✅ 代码生成日志系统已初始化")

    def _init_llm_client(self):
        """
        初始化LLM客户端 - 使用TableZoomer的配置文件
        """
        try:
            # 导入weaver的核心组件
            weaver_path = Path("/home/lilin/weaver")
            if weaver_path.exists() and str(weaver_path) not in sys.path:
                sys.path.insert(0, str(weaver_path))
            
            # 正确导入Weaver的LLM客户端和配置系统
            from weaver.llm.client import LLMClient
            from weaver.config.settings import WeaverConfig, LLMConfig
            
            # 加载TableZoomer的配置文件
            config_path = Path("/home/lilin/TableZoomer/agent_config/weaver_config.yaml")
            weaver_config = WeaverConfig.from_env()
            
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
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
                        weaver_config.llm.api_base = llm_config['base_url']
                    if 'api_key' in llm_config:
                        # 设置API密钥到环境变量
                        os.environ['OPENAI_API_KEY'] = llm_config['api_key']
                
                logger.info("✅ 成功加载TableZoomer配置文件")
            else:
                logger.warning("⚠️ TableZoomer配置文件不存在，使用默认配置")
            
            # 创建LLM客户端实例
            self.llm_client = LLMClient(weaver_config.llm)
            logger.info("✅ 成功加载weaver LLM客户端")
            logger.info(f"   使用模型: {weaver_config.llm.model}")
            logger.info(f"   API Base: {weaver_config.llm.api_base}")
            logger.info(f"   温度: {weaver_config.llm.temperature}")
            logger.info(f"   最大token: {weaver_config.llm.max_tokens}")
        except Exception as e:
            # 直接抛出异常，不再使用备用方案
            logger.error(f"❌ 加载weaver LLM客户端失败: {str(e)}")
            logger.error("请确保weaver库已正确安装且配置文件存在")
            import traceback
            traceback.print_exc()
            raise ImportError(f"无法加载weaver LLM客户端: {str(e)}") from e

    def _init_database(self):
        """
        初始化数据库管理器
        """
        try:
            # 导入weaver的数据库管理器
            weaver_path = Path("/home/lilin/weaver")
            if weaver_path.exists() and str(weaver_path) not in sys.path:
                sys.path.insert(0, str(weaver_path))
            
            from weaver.database.manager import DatabaseManager
            from weaver.config.settings import WeaverConfig
            
            # 加载TableZoomer的配置文件
            config_path = Path("/home/lilin/TableZoomer/agent_config/weaver_config.yaml")
            weaver_config = WeaverConfig.from_env()
            
            if config_path.exists():
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
                
                logger.info("✅ 成功加载数据库配置")
            
            # 创建数据库管理器实例
            self.database = DatabaseManager(weaver_config.database)
            logger.info("✅ 成功初始化数据库管理器")
            logger.info(f"   数据库类型: {weaver_config.database.db_type}")
            logger.info(f"   数据库路径: {weaver_config.database.db_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ 初始化数据库管理器失败: {str(e)}")
            logger.warning("将使用内存中的临时数据库")
            # 使用内存数据库作为备用方案
            try:
                import duckdb
                self.database = duckdb.connect(":memory:")
                logger.info("✅ 使用内存数据库作为备用方案")
            except Exception as duckdb_e:
                logger.error(f"❌ 备用数据库初始化也失败: {str(duckdb_e)}")
                self.database = None

    def _load_prompt(self, filename):
        """
        加载提示词文件
        """
        prompt_file = self.prompts_dir / filename
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"✅ 加载提示词文件: {filename}，长度: {len(content)}")
                return content
        else:
            logger.warning(f"⚠️ 提示词文件不存在: {prompt_file}")
            return ""

    def _load_table_data(self, query=None):
        """
        加载表格数据并使用Weaver的预处理逻辑进行清洗
        """
        try:
            if self.table_desc and 'file_path' in self.table_desc:
                file_path = self.table_desc['file_path']
                logger.info(f"加载表格文件: {file_path}")
                
                # 直接读取文件
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, encoding='utf8', on_bad_lines='skip')
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    return None
                
                if df is not None and not df.empty:
                    logger.info(f"成功加载表格数据，共 {len(df)} 行，列数: {len(df.columns)}")
                    logger.info(f"原始表格列名: {list(df.columns)}")
                    
                    # 使用Weaver的TablePreprocessor进行表格清洗
                    logger.info("使用Weaver的表格预处理逻辑清洗表格数据...")
                    preprocessor = TablePreprocessor()
                    
                    # 优先使用table_desc中的clean_table_name（如果存在）
                    if hasattr(self, 'table_desc') and 'clean_table_name' in self.table_desc:
                        table_name = self.table_desc['clean_table_name']
                        logger.info(f"使用table_desc中的clean_table_name: {table_name}")
                    else:
                        table_name = os.path.basename(file_path).split('.')[0]
                        logger.info(f"使用文件名生成table_name: {table_name}")
                        
                    clean_table_name, clean_df = preprocessor.clean_table(df, table_name)
                    logger.info(f"表格清洗完成，新表名: {clean_table_name}")
                    logger.info(f"清洗后表格列名: {list(clean_df.columns)}")
                    logger.info(f"清洗后表格大小: {clean_df.shape}")
                    
                    # 将clean_table_name存储到DataFrame的attrs属性中，供后续SQL查询使用
                    if not hasattr(clean_df, 'attrs'):
                        clean_df.attrs = {}
                    clean_df.attrs['clean_table_name'] = clean_table_name
                    logger.info(f"✅ 已将clean_table_name存储到DataFrame的attrs属性: {clean_table_name}")
                    
                    # 上传清洗后的表格到数据库 - 使用clean_table_name而不是硬编码的"current_table"
                    if self.database is not None:
                        try:
                            # 检查数据库管理器类型
                            if hasattr(self.database, 'upload_table'):
                                # 使用weaver的DatabaseManager
                                success = self.database.upload_table(clean_table_name, clean_df, if_exists="replace")
                                if success:
                                    logger.info(f"✅ 成功将清洗后的表格上传到数据库: {clean_table_name}")
                                else:
                                    logger.warning("⚠️ 表格上传失败")
                            else:
                                # 使用duckdb直接操作
                                self.database.register(clean_table_name, clean_df)
                                logger.info(f"✅ 成功将清洗后的表格注册到内存数据库: {clean_table_name}")
                        except Exception as upload_error:
                            logger.warning(f"⚠️ 表格上传失败: {str(upload_error)}")
                    
                    return clean_df
                else:
                    logger.warning("表格数据为空")
                    return None
            else:
                logger.warning("未找到文件路径，无法加载表格数据")
                return None
        except Exception as e:
            logger.error(f"加载表格数据失败: {str(e)}")
            return None

    def call_llm(self, prompt):
        """
        调用LLM生成响应（使用Weaver的LLM客户端）
        """
        try:
            logger.info(f"调用LLM，提示词长度: {len(prompt)}")
            
            if not hasattr(self.llm_client, 'call'):
                logger.error("❌ LLM客户端不具有call方法")
                raise AttributeError("LLM客户端不具有call方法")
            
            # 直接使用Weaver的LLM客户端
            response = self.llm_client.call(prompt)
            logger.info(f"LLM返回成功，响应长度: {len(response)}")
            
            # 改为使用INFO级别，确保在终端中显示完整响应
            logger.info("=== LLM完整响应 ===")
            logger.info(response)
            logger.info("===================")
            
            return response.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def generate_plan(self, query, table_schema, paragraph_schema):
        """
        基于TableZoomer提示词生成执行计划
        """
        logger.info("=== 生成执行计划 ===")
        logger.info(f"查询: {query}")
        logger.info(f"表格schema: {table_schema}")
        logger.info(f"段落schema: {paragraph_schema}")
        
        if not self.planner_prompt:
            logger.error("计划生成提示词未加载")
            return None
        
        # 构建完整的提示词
        full_prompt = self.planner_prompt.replace("{query}", query)
        full_prompt = full_prompt.replace("{table_schema}", str(table_schema))
        full_prompt = full_prompt.replace("{paragraph_schema}", str(paragraph_schema))
        
        logger.info(f"计划生成提示词构建完成，长度: {len(full_prompt)}")
        
        # 调用LLM生成计划
        plan_text = self.call_llm(full_prompt)
        
        # 确保生成的计划在终端中完整显示
        logger.info("=== 生成的执行计划 ===")
        logger.info(plan_text)
        logger.info("=====================")
        
        return plan_text

    def verify_plan(self, query, table_schema, paragraph_schema, plan_text):
        """
        验证生成的执行计划
        """
        logger.info("=== 验证执行计划 ===")
        logger.info(f"原始计划: {plan_text}")
        
        if not self.verify_prompt:
            logger.error("计划验证提示词未加载")
            return True
        
        # 构建验证提示词
        full_prompt = self.verify_prompt.replace("{query}", query)
        full_prompt = full_prompt.replace("{table_schema}", str(table_schema))
        full_prompt = full_prompt.replace("{paragraph_schema}", str(paragraph_schema))
        full_prompt = full_prompt.replace("{plan}", plan_text)
        
        logger.info(f"计划验证提示词构建完成，长度: {len(full_prompt)}")
        
        # 调用LLM验证计划
        verify_result = self.call_llm(full_prompt)
        
        # 确保验证结果在终端中完整显示
        logger.info("=== 计划验证结果 ===")
        logger.info(verify_result)
        logger.info("===================")
        
        # 判断验证结果 - 同时检查中文和英文的验证通过关键词
        success_keywords = [
            "有效", "有效的", "VALID", 
            "Plan verified", "No changes needed", "✅"
        ]
        
        # 检查是否包含任何验证通过的关键词
        for keyword in success_keywords:
            if keyword in verify_result:
                logger.info("✅ 执行计划验证通过")
                return True
        
        # 如果没有找到任何验证通过的关键词，则验证失败
        logger.warning("❌ 执行计划验证失败")
        return False

    def parse_plan_steps(self, plan_text):
        """
        解析执行计划文本，提取步骤
        """
        logger.info("=== 解析执行计划步骤 ===")
        logger.info(f"待解析的计划文本: {plan_text}")
        
        steps = []
        
        # 使用正则表达式匹配步骤
        # 匹配格式如: 1. [Tool] 描述 或 1.[Tool]描述
        pattern = r'(\d+)\.\s*\[([^\]]+)\]\s*(.*)'
        matches = re.findall(pattern, plan_text)
        
        if not matches:
            logger.warning("❌ 未找到有效的执行步骤，使用默认执行步骤")
            # 默认执行步骤
            steps.append({
                "step": 1,
                "tool": "QueryTool",
                "description": "直接查询表格数据"
            })
        else:
            for match in matches:
                step_num = int(match[0])
                tool = match[1]
                description = match[2].strip()
                
                steps.append({
                    "step": step_num,
                    "tool": tool,
                    "description": description
                })
            
            logger.info(f"✅ 成功解析 {len(steps)} 个执行步骤")
            for step in steps:
                logger.info(f"   步骤 {step['step']}: [{step['tool']}] {step['description']}")
        
        return steps

    def _execute_sql_query(self, table_name, df, step_description, query, previous_results=None):
        """
        执行SQL查询
        """
        try:
            # 根据步骤描述生成SQL查询
            sql_query = self._generate_sql_from_description(table_name, df, step_description, query, previous_results)
            
            logger.info(f"执行SQL查询: {sql_query}")
            
            if self.database is not None:
                # 执行查询
                if hasattr(self.database, 'execute_query'):
                    # 使用weaver的DatabaseManager
                    result = self.database.execute_query(sql_query)
                    if result.success:
                        if result.data is not None and not result.data.empty:
                            return f"查询结果: {result.data.to_dict()}"
                        else:
                            return "查询成功，但无数据返回"
                    else:
                        return f"查询失败: {result.error}"
                else:
                    # 使用duckdb直接执行
                    result = self.database.execute(sql_query).fetchall()
                    if result:
                        return f"查询结果: {result}"
                    else:
                        return "查询成功，但无数据返回"
            else:
                return "数据库未初始化，无法执行查询"
                
        except Exception as e:
            logger.error(f"SQL查询执行失败: {str(e)}")
            return f"SQL查询失败: {str(e)}"

    def _generate_sql_from_description(self, table_name, df, step_description, query, previous_results=None):
        """
        根据步骤描述生成SQL查询
        """
        # 使用清洗后的实际列名构建schema，而不是df.dtypes.to_dict()
        table_schema = []
        for col in df.columns:
            table_schema.append({
                "name": col,
                "type": str(df[col].dtype)
            })
        
        logger.info(f"使用清洗后的列名构建SQL查询schema: {[col['name'] for col in table_schema]}")
        
        # 构建完整提示词，包含之前的执行结果
        prompt = f"""
Generate SQL query based on the following information:

Table Name: {table_name}
Table Schema: {table_schema}
Step Description: {step_description}
User Query: {query}

Previous Execution Results: {previous_results}

First, analyze and interpret the previous execution results to understand what information they contain.
Then, generate a SQL query that builds upon this information to achieve the current step's goal.

Please generate only the SQL query without any explanation.
"""
        
        # 调用LLM生成SQL
        sql_query = self.call_llm(prompt)
        
        # 提取SQL代码并记录到代码生成日志
        sql_query = self._extract_sql_code(sql_query, prompt)
        
        return sql_query        

    def _extract_sql_code(self, response, prompt=None):
        """
        从LLM响应中提取SQL代码并记录日志
        """
        # 查找SQL代码块
        sql_pattern = r"```sql(.*?)```"
        match = re.search(sql_pattern, response, re.DOTALL)
        if match:
            sql_code = match.group(1).strip()
        else:
            # 如果没有找到代码块，尝试提取纯SQL
            sql_code = response.strip()
            # 确保SQL以SELECT开头
            if not sql_code.upper().startswith("SELECT"):
                logger.warning(f"生成的SQL可能不完整: {sql_code}")
        
        # 记录代码生成日志
        log_code_generation("SQL代码生成", "SQL", sql_code, prompt)
        
        return sql_code

    def _execute_python_code(self, code_description, query, table_schema, previous_results=None):
        """
        Generate and execute Python code by LLM itself
        """
        try:
            # 生成Python代码
            python_code = self._generate_python_code(code_description, query, table_schema, previous_results)
            
            logger.info(f"Generated Python code: {python_code}")
            
            # Let LLM execute the code and return results
            execution_prompt = f"""
Please execute the following Python code and return the result:

```python
{python_code}

Previous execution results that might be relevant:
{previous_results}

Please provide your response in the following JSON format:
{{
  "execution_status": "success" or "failed",
  "execution_result": the result of the code execution
}}

Make sure to compute the result accurately. If there's an error in execution, please explain the error in the execution_result field.
"""
            
            # Call LLM to execute the code
            llm_response = self.call_llm(execution_prompt)
            
            logger.info(f"LLM execution response: {llm_response}")
            
            # 记录Python代码执行日志
            log_code_generation("Python代码执行", "Python", python_code, execution_prompt, llm_response)
            
            # Parse the JSON response
            try:
                result_dict = json.loads(llm_response)
                execution_status = result_dict.get("execution_status", "failed")
                execution_result = result_dict.get("execution_result", "No result provided")
                
                # Return the result in the expected format
                if execution_status == "success":
                    return execution_result
                else:
                    logger.error(f"Python code execution failed according to LLM: {execution_result}")
                    return f"Python code execution failed: {execution_result}"
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response with error
                logger.error(f"Failed to parse LLM execution response as JSON: {llm_response}")
                return f"Python code execution failed: Failed to parse execution result"
        except Exception as e:
            logger.error(f"Python code execution failed: {str(e)}")
            return f"Python code execution failed: {str(e)}"

    def _generate_python_code(self, code_description, query, table_schema, previous_results=None):
        """
        根据描述生成Python代码
        """
        # 构建完整提示词
        prompt = f"""
Generate Python code based on the following information:

Code Description: {code_description}
User Query: {query}
Table Schema: {table_schema}

Previous Execution Results: {previous_results}

First, analyze and interpret the previous execution results to understand what information they contain.
Then, generate Python code that builds upon this information to achieve the current step's goal.

Please generate only the Python code without any explanation.
Make sure the code is self-contained and can be executed directly.
"""
        # 调用LLM生成Python代码
        python_code = self.call_llm(prompt)
        
        # 提取Python代码并记录日志
        python_code = self._extract_python_code(python_code, prompt)
        
        return python_code

    def _extract_python_code(self, response, prompt=None):
        """
        从LLM响应中提取Python代码并记录日志
        """
        # 查找Python代码块
        python_pattern = r"```python(.*?)```"
        match = re.search(python_pattern, response, re.DOTALL)
        if match:
            python_code = match.group(1).strip()
        else:
            # 如果没有找到代码块，尝试提取纯Python
            python_code = response.strip()
        
        # 记录代码生成日志
        log_code_generation("Python代码生成", "Python", python_code, prompt)
        
        return python_code

    def execute_step(self, step, df, query, previous_results=None):
        """
        执行单个计划步骤 - 实现生成一步代码→执行一步的逻辑
        """
        logger.info(f"=== 执行步骤 {step['step']} ===")
        logger.info(f"工具: {step['tool']}")
        logger.info(f"描述: {step['description']}")
        logger.info(f"之前的执行结果: {previous_results}")
        
        try:
            if step['tool'] == "SQL" or "sql" in step['tool'].lower():
                # 执行SQL查询
                logger.info("执行SQL查询...")
                
                if df is not None:
                    # 获取表格名称（优先使用df的attrs中的clean_table_name，如果没有则使用默认名称）
                    table_name = None
                    if hasattr(df, 'attrs') and 'clean_table_name' in df.attrs:
                        table_name = df.attrs['clean_table_name']
                    
                    # 确保table_name有值
                    if not table_name:
                        # 这里可以使用一个默认的表名，但根据您的需求，我们应该确保table_name已经设置
                        table_name = "table_1"
                        logger.warning(f"未找到clean_table_name，使用默认表名: {table_name}")
                    
                    # 执行SQL查询，传入之前的执行结果
                    sql_result = self._execute_sql_query(table_name, df, step['description'], query, previous_results)
                    
                    return {
                        "success": True,
                        "result": sql_result
                    }
                else:
                    return {
                        "success": False,
                        "error": "表格数据为空"
                    }
            elif step['tool'] == "Python" or "python" in step['tool'].lower():
                # 执行Python代码
                logger.info("执行Python代码...")
                
                # 使用清洗后的实际列名构建schema，而不是df.dtypes.to_dict()
                table_schema = []
                for col in df.columns:
                    table_schema.append({
                        "name": col,
                        "type": str(df[col].dtype)
                    })
                
                # 生成并执行Python代码，传入之前的执行结果
                python_result = self._execute_python_code(step['description'], query, table_schema, previous_results)
                
                return {
                    "success": True,
                    "result": python_result
                }
            elif step['tool'] == "LLM" or "llm" in step['tool'].lower():
                # 执行LLM推理
                logger.info("执行LLM推理...")
                
                # 构建推理提示词
                prompt = f"""
Based on the following information, provide your reasoning:

Step Description: {step['description']}
User Query: {query}
Table Schema: {df.dtypes.to_dict()}
Previous Results: {previous_results}

Please provide a clear and concise reasoning or answer.
"""
                
                # 调用LLM生成推理结果
                llm_result = self.call_llm(prompt)
                
                return {
                    "success": True,
                    "result": llm_result
                }
            #
            elif step['tool'] == "QueryTool" or "查询" in step['tool']:
                # 执行查询工具（兼容旧格式）
                logger.info("执行查询工具...")
                
                if df is not None:
                    # 获取表格名称 - 优先使用df的attrs中的clean_table_name
                    table_name = None
                    if hasattr(df, 'attrs') and 'clean_table_name' in df.attrs:
                        table_name = df.attrs['clean_table_name']
                    
                    # 确保table_name有值
                    if not table_name:
                        # 如果仍然没有表名，使用默认名称
                        table_name = "table_1"
                        logger.warning(f"未找到clean_table_name，使用默认表名: {table_name}")
                    
                    # 执行SQL查询，传入之前的执行结果
                    sql_result = self._execute_sql_query(table_name, df, step['description'], query, previous_results)
                    
                    return {
                        "success": True,
                        "result": sql_result
                    }
                else:
                    return {
                        "success": False,
                        "error": "表格数据为空"
                    }
            else:
                # 未知工具，直接返回成功
                logger.warning(f"未知工具: {step['tool']}，直接返回成功")
                return {
                    "success": True,
                    "result": f"执行 {step['tool']} 完成"
                }
        except Exception as e:
            logger.error(f"执行步骤失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _execute_fallback_query(self, df, query):
        """
        备用查询方式 - 当数据库上传失败时使用
        """
        logger.info("使用备用查询方式...")
        
        try:
            # 简单的pandas查询
            if "计数" in query or "多少" in query:
                result = f"表格包含 {len(df)} 行数据"
            elif "列" in query:
                result = f"表格列名: {', '.join(list(df.columns))}"
            else:
                # 显示前几行数据
                sample_data = df.head(3).to_dict('records')
                result = f"表格样本数据: {sample_data}"
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"备用查询失败: {str(e)}"
            }

    def extract_answer(self, query, table_schema, paragraph_schema, execution_results):
        """
        从执行结果中提取最终答案
        """
        logger.info("=== 提取最终答案 ===")
        
        # 构建提取答案的提示词
        if not self.extract_answer_prompt:
            logger.error("答案提取提示词未加载")
            return "无法提取答案，提示词未加载"
        
        # 收集所有执行结果
        results_text = "\n".join([
            f"步骤 {res['step']}: {res['result']}"
            for res in execution_results if res['success']
        ])
        
        if not results_text:
            results_text = "没有有效的执行结果"
        
        # 构建完整提示词
        full_prompt = self.extract_answer_prompt.replace("{query}", query)
        full_prompt = full_prompt.replace("{table_schema}", str(table_schema))
        full_prompt = full_prompt.replace("{paragraph_schema}", str(paragraph_schema))
        full_prompt = full_prompt.replace("{execution_results}", results_text)
        
        logger.info(f"答案提取提示词构建完成，长度: {len(full_prompt)}")
        
        # 调用LLM提取答案
        answer = self.call_llm(full_prompt)
        
        # 确保答案在终端中完整显示
        logger.info("=== 最终答案 ===")
        logger.info(answer)
        logger.info("=============")
        
        return answer

    def process_question(self, query, table_schema, table_desc=None, paragraph_schema=None):
        """
        处理问题并返回答案
        实现生成一步代码→执行一步→根据结果生成下一步代码的完整流程
        """
        logger.info("=== 开始处理问题 ===")
        logger.info(f"查询: {query}")
        logger.info(f"表格schema: {table_schema}")
        logger.info(f"表格描述: {table_desc}")
        logger.info(f"段落schema: {paragraph_schema}")
        
        try:
            # 1. 加载表格数据（使用Weaver的预处理逻辑）
            self.table_desc = table_desc
            
            df = self._load_table_data(query)
            
            # 确保df包含clean_table_name属性
            if df is not None:
                # 检查是否已经设置了clean_table_name
                if not hasattr(df, 'attrs') or 'clean_table_name' not in df.attrs:
                    logger.warning("⚠️ DataFrame中没有clean_table_name属性，尝试从文件名推断")
                    # 尝试从table_desc中获取表名
                    if table_desc and 'file_path' in table_desc:
                        file_path = table_desc['file_path']
                        table_name = os.path.basename(file_path).split('.')[0]
                        if not hasattr(df, 'attrs'):
                            df.attrs = {}
                        df.attrs['clean_table_name'] = table_name
                        logger.info(f"✅ 已从文件名推断并设置clean_table_name: {table_name}")
            
            # 2. 如果没有提供table_schema，基于清洗后的表格生成schema
            if not table_schema and df is not None:
                table_schema = []
                for col in df.columns:
                    # 获取列的示例值
                    sample_values = df[col].dropna().head(3).tolist()
                    table_schema.append({
                        "name": col,
                        "type": str(df[col].dtype),
                        "sample_values": sample_values
                    })
            
            # 3. 如果没有提供paragraph_schema，尝试从table_desc中获取
            if not paragraph_schema and table_desc and "paragraphs" in table_desc:
                paragraph_schema = []
                for i, para in enumerate(table_desc["paragraphs"]):
                    paragraph_schema.append({
                        "id": i + 1,
                        "content": para[:100] + "..." if len(para) > 100 else para
                    })
            
            # 4. 生成执行计划
            plan_text = self.generate_plan(query, table_schema, paragraph_schema)
            
            # 5. 验证执行计划
            is_valid = self.verify_plan(query, table_schema, paragraph_schema, plan_text)
            logger.info(f"计划验证结果: {'有效' if is_valid else '无效'}")
            
            # 6. 解析执行计划步骤
            steps = self.parse_plan_steps(plan_text)
            
            # 7. 执行计划步骤 - 实现生成一步代码→执行一步→根据结果生成下一步代码的流程
            execution_results = []
            previous_results = None
            
            for step in steps:
                # 执行当前步骤，传入之前的执行结果
                result = self.execute_step(step, df, query, previous_results)
                
                # 记录执行结果
                execution_results.append({
                    "step": step['step'],
                    "tool": step['tool'],
                    "result": result['result'] if result['success'] else f"失败: {result['error']}",
                    "success": result['success']
                })
                
                # 更新之前的执行结果，用于下一步
                if result['success']:
                    previous_results = result['result']
                else:
                    logger.error(f"步骤 {step['step']} 执行失败，终止执行")
                    break
            
            # 8. 提取最终答案
            answer = self.extract_answer(query, table_schema, paragraph_schema, execution_results)
            
            logger.info("=== 问题处理完成 ===")
            logger.info(f"最终答案: {answer}")
            
            return {
                "answer": answer,
                "success": True,
                "execute_state": "success",
                "plan": plan_text,
                "steps": steps,
                "execution_results": execution_results
            }
        except Exception as e:
            logger.error(f"❌ 处理问题失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试备用执行方式
            try:
                logger.info("尝试备用执行方式...")
                if df is not None:
                    # 简单直接查询
                    logger.info("直接查询表格数据...")
                    answer = f"表格包含 {len(df)} 行数据，列包括: {', '.join(list(df.columns))}"
                    return {
                        "answer": answer,
                        "success": True,
                        "execute_state": "fallback"
                    }
            except Exception as fallback_e:
                logger.error(f"备用执行失败: {str(fallback_e)}")
            
            return {
                "answer": f"Error: {str(e)}",
                "success": False,
                "execute_state": "fail"
            }

# 保持向后兼容性
WeaverBasedCodeExecutor = TableZoomerCodeExecutor