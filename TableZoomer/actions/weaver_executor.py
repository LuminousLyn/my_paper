"""
TableZoomer的Weaver代码生成执行器
用于复用Weaver的plan-execute框架实现SQL和POT(Program of Thoughts)的交替使用
"""

# 修复模块导入问题
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 添加 weaver 模块路径

from typing import Dict, Any, Optional
import pandas as pd
import json
import logging
import subprocess
import ast
import re
import asyncio
from metagpt.logs import logger

from weaver.core.weaver import TableQA
from weaver.config.settings import WeaverConfig
from weaver.database.manager import DatabaseManager
from weaver.prompts import load_prompt, configure_prompt_loader
from weaver.llm.client import create_llm_client

# 定义 execution_types
execution_types = ["LLM", "SQL", "PoT"]  # 默认支持的执行类型

class WeaverCodeGenerator:
    """使用Weaver的plan-execute框架生成代码（轻量版，不依赖MetaGPT Action初始化）"""
    def __init__(self, config: Optional[WeaverConfig] = None):
        """初始化代码生成器

        This implementation avoids inheriting from MetaGPT Action to prevent
        pydantic/ContextMixin initialization issues. It constructs its own
        LLM client using Weaver's LLM wrapper.
        """
        if config is None:
            config = WeaverConfig.from_env()

        self.config = config
        # configure prompt loader to use external prompts dir if provided
        try:
            configure_prompt_loader(self.config.prompts_dir)
        except Exception:
            # non-fatal
            pass

        # LLM client (synchronous), we will call it via asyncio.to_thread
        self.llm = create_llm_client(self.config.llm)

        self._setup_database()
    
    def _setup_database(self):
        """初始化数据库连接"""
        conn_str = self.config.database.get_connection_string()
        db_type = self.config.database.db_type
        self.database = DatabaseManager(conn_str, db_type)
        logger.info(f"Database initialized: {db_type} - {conn_str}")
        
    async def run(self, instruction: str) -> str:
        """生成代码
        
        Args:
            instruction: 包含查询和表格信息的JSON字符串
            
        Returns:
            生成的代码和执行计划
        """
        instruction = json.loads(instruction)
        query = instruction['query']
        table_desc = instruction.get('table_desc', {})
        table_zoom = instruction.get('table_zoom')
        
        # 1. 生成执行计划
        plan = await self._create_plan(query, table_desc, table_zoom)
        logger.info(f"Generated plan:\n{plan}")
        
        # 2. 生成代码
        code = await self._generate_code(plan, query, table_desc, table_zoom)
        logger.info(f"Generated code:\n{code}")
        
        # 3. 构造返回结果
        result = {
            "code": code,
            "plan": plan,
            "table_desc": table_desc
        }
        
        return json.dumps({"prompt": instruction, "rsp": result}, ensure_ascii=False)
        
    async def _create_plan(self, query: str, table_desc: Dict[str, Any], table_zoom: Optional[str] = None, paragraph_schema: Optional[Dict[str, Any]] = None) -> str:
        """创建执行计划，支持 paragraph_schema
        
        Args:
            query: 用户查询
            table_desc: 表格描述
            table_zoom: 压缩的表格信息(可选)
            paragraph_schema: 段落模式下的表格结构描述（可选）
            
        Returns:
            执行计划字符串
        """
        # 加载plan prompt
        prompt = load_prompt("planner_prompt", getattr(self.config, "default_dataset", "default"))

        # 构建完整prompt
        full_prompt = prompt + f"""

Table Schema:
{json.dumps(table_desc, indent=2)}

Query: {query}

"""
        if table_zoom:
            full_prompt += f"""
Additional Table Information (Table Zoom):
{table_zoom}
"""

        if paragraph_schema:
            full_prompt += f"""
Paragraph Schema:
{json.dumps(paragraph_schema, indent=2)}
"""

        # 调用LLM生成计划（weaver LLM client is sync, run in thread）
        plan = await asyncio.to_thread(self.llm.call, full_prompt)
        return plan.strip()
        
    async def _generate_code(self, 
                           plan: str,
                           query: str, 
                           table_desc: Dict[str, Any],
                           table_zoom: Optional[str] = None) -> str:
        """根据计划生成代码
        
        Args:
            plan: 执行计划
            query: 用户查询
            table_desc: 表格描述
            table_zoom: 压缩的表格信息(可选)
            
        Returns:
            生成的代码
        """
        # 加载代码生成prompt
        prompt = load_prompt("execute_prompt", getattr(self.config, "default_dataset", "default"))

        # 构建完整prompt
        full_prompt = prompt + f"""

Table Schema:
{json.dumps(table_desc, indent=2)}

Query: {query}

Execution Plan:
{plan}

"""
        if table_zoom:
            full_prompt += f"""
Additional Table Information (Table Zoom):
{table_zoom}
"""

        # 调用LLM生成代码
        code = await asyncio.to_thread(self.llm.call, full_prompt)
        return code.strip()


class WeaverCodeExecutor:
    """使用Weaver的执行引擎运行生成的代码（轻量版，不依赖MetaGPT Action）"""
    def __init__(self, config: Optional[WeaverConfig] = None):
        """初始化代码执行器"""
        if config is None:
            config = WeaverConfig.from_env()

        self.config = config
        # LLM client may not be needed here, but keep for parity
        self.llm = create_llm_client(self.config.llm)
        # configure prompt loader for executor too
        try:
            configure_prompt_loader(self.config.prompts_dir)
        except Exception:
            pass

        self._setup_database()
        
    def _setup_database(self):
        """初始化数据库连接"""
        conn_str = self.config.database.get_connection_string()
        db_type = self.config.database.db_type
        self.database = DatabaseManager(conn_str, db_type)
        logger.info(f"Database initialized: {db_type} - {conn_str}")
    
    async def run(self, inputs: str):
        """执行代码
        
        Args:
            inputs: 包含代码的JSON字符串
            
        Returns:
            执行结果
        """
        try:
            # 解析输入
            inputs_dict = json.loads(inputs)
            code_rsp = inputs_dict['rsp']
            code_gen_prompt = inputs_dict['prompt']
            
            try:
                code_instructions = (
                    json.loads(code_rsp) if isinstance(code_rsp, str)
                    else code_rsp
                )
            except Exception as e:
                logger.warning(f'code_rsp load warning: {e}. Try to use ast.literal_eval()')
                try:
                    code_instructions = ast.literal_eval(code_rsp)
                except Exception as e:
                    logger.warning(f'code_rsp parse failed: {e}')
                    return json.dumps({
                        "prompt": code_gen_prompt,
                        "code_rsp": '',
                        "code": '',
                        "response": '',
                        "execute_state": 'fail',
                        "error": f"Code parsing failed: {e}"
                    }, ensure_ascii=False)
            
            code = code_instructions['code']
            plan = code_instructions.get('plan', '')
            
            # 解析SQL和Python代码
            code_blocks = self._parse_code_blocks(code)
            
            # 执行每个代码块
            final_result = ''
            current_df = None
            
            for block in code_blocks:
                block_type = block['type']
                block_code = block['code']
                
                if block_type == 'sql':
                    # 执行SQL
                    result = self.database.execute_query(block_code)
                    if not result.success:
                        return json.dumps({
                            "prompt": code_gen_prompt,
                            "code_rsp": code_rsp,
                            "code": code,
                            "response": '',
                            "execute_state": 'fail',
                            "error": f"SQL execution failed: {result.error}"
                        }, ensure_ascii=False)
                    current_df = result.data
                    
                elif block_type == 'python':
                    # 如果有DataFrame结果,注入到Python环境
                    if current_df is not None:
                        block_code = f"""
import pandas as pd
df = pd.DataFrame({current_df.to_dict()})
{block_code}
"""
                    # 执行Python代码
                    try:
                        result = subprocess.run(
                            ["python3", "-c", block_code],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=60
                        )
                        if result.returncode != 0:
                            return json.dumps({
                                "prompt": code_gen_prompt,
                                "code_rsp": code_rsp,
                                "code": code,
                                "response": '',
                                "execute_state": 'fail',
                                "error": result.stderr
                            }, ensure_ascii=False)
                        execution_output = result.stdout.strip()
                        if execution_output:
                            final_result = execution_output
                            
                    except Exception as e:
                        return json.dumps({
                            "prompt": code_gen_prompt,
                            "code_rsp": code_rsp,
                            "code": code,
                            "response": '',
                            "execute_state": 'fail',
                            "error": str(e)
                        }, ensure_ascii=False)
            
            # 返回成功结果
            return json.dumps({
                "prompt": code_gen_prompt,
                "code_rsp": code_rsp,
                "code": code,
                "response": final_result,
                "execute_state": 0,
                "plan": plan
            }, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "prompt": code_gen_prompt if 'code_gen_prompt' in locals() else '',
                "code_rsp": '',
                "code": '',
                "response": '',
                "execute_state": 'fail',
                "error": str(e)
            }, ensure_ascii=False)
            
    def _parse_code_blocks(self, code: str) -> list:
        """解析代码中的SQL和Python块
        
        Args:
            code: 完整代码字符串
            
        Returns:
            代码块列表,每个块包含type和code
        """
        blocks = []
        
        # 首先尝试根据执行类型列表解析
        if execution_types:
            # 提取SQL块
            sql_pattern = r"(?:--\s*SQL.*?\n)(.*?)(?=--\s*(?:SQL|Python|LLM)|$)"
            sql_matches = re.finditer(sql_pattern, code, re.DOTALL)
            for match in sql_matches:
                sql_code = match.group(1).strip()
                if sql_code:
                    blocks.append({
                        "type": "sql",
                        "code": sql_code
                    })
                    
            # 提取Python块
            python_pattern = r"(?:--\s*Python.*?\n)(.*?)(?=--\s*(?:SQL|Python|LLM)|$)"
            python_matches = re.finditer(python_pattern, code, re.DOTALL)
            for match in python_matches:
                python_code = match.group(1).strip()
                if python_code:
                    blocks.append({
                        "type": "python",
                        "code": python_code
                    })
                    
            # 提取LLM块（如果存在）
            if "LLM" in execution_types:
                # 查找可能的LLM指令部分
                llm_pattern = r"(?:--\s*LLM.*?\n)(.*?)(?=--\s*(?:SQL|Python|LLM)|$)"
                llm_matches = re.finditer(llm_pattern, code, re.DOTALL)
                for match in llm_matches:
                    llm_code = match.group(1).strip()
                    if llm_code:
                        blocks.append({
                            "type": "llm",
                            "code": llm_code
                        })
        
        # 如果没有找到块，使用默认的解析方式
        if not blocks:
            # 使用更可靠的解析逻辑
            # 提取SQL块
            sql_pattern = r"--\s*SQL.*?\n(.*?)(?=--\s*(?:SQL|Python|LLM|$))"
            sql_matches = re.finditer(sql_pattern, code, re.DOTALL | re.IGNORECASE)
            for match in sql_matches:
                sql_code = match.group(1).strip()
                if sql_code:
                    blocks.append({"type": "sql", "code": sql_code})
                    
            # 提取Python块
            python_pattern = r"--\s*Python.*?\n(.*?)(?=--\s*(?:SQL|Python|LLM|$))"
            python_matches = re.finditer(python_pattern, code, re.DOTALL | re.IGNORECASE)
            for match in python_matches:
                python_code = match.group(1).strip()
                if python_code:
                    blocks.append({"type": "python", "code": python_code})
            
            # 提取LLM块
            llm_pattern = r"--\s*LLM.*?\n(.*?)(?=--\s*(?:SQL|Python|LLM|$))"
            llm_matches = re.finditer(llm_pattern, code, re.DOTALL | re.IGNORECASE)
            for match in llm_matches:
                llm_code = match.group(1).strip()
                if llm_code:
                    blocks.append({"type": "llm", "code": llm_code})
        
        # 如果还是没有找到块，尝试最基本的解析方式
        if not blocks:
            # 查找任何SQL语句
            sql_simple_pattern = r"(SELECT|INSERT|UPDATE|DELETE).*?;"
            sql_simple_matches = re.finditer(sql_simple_pattern, code, re.DOTALL | re.IGNORECASE)
            for match in sql_simple_matches:
                sql_code = match.group(0).strip()
                if sql_code:
                    blocks.append({"type": "sql", "code": sql_code})
            
            # 查找任何Python代码块（简单启发式）
            if "# Input:" in code or "import pandas" in code or "df." in code:
                blocks.append({"type": "python", "code": code})
        
        return sorted(blocks, key=lambda x: code.find(x['code']))  # 按原始顺序排序


class TableZoomerExecutor:
    """Compatibility wrapper expected by TableZoomer code.

    Provides a simple synchronous `execute_query` method that uses the
    WeaverCodeGenerator and WeaverCodeExecutor defined above.
    """
    def __init__(self, config: Optional[WeaverConfig] = None):
        # accept dict-like config for backwards compatibility
        if isinstance(config, dict):
            # create a WeaverConfig from env and override known fields
            wc = WeaverConfig.from_env()
            # override llm.model if provided
            llm_conf = config.get('llm') or config.get('LLM')
            if isinstance(llm_conf, dict) and llm_conf.get('model'):
                wc.llm.model = llm_conf.get('model')
            # override dataset/prompts/results dirs if provided
            prompts = config.get('prompts') or {}
            if isinstance(prompts, dict) and prompts.get('dir'):
                wc.prompts_dir = Path(prompts.get('dir'))
            data = config.get('data') or {}
            if isinstance(data, dict) and data.get('max_table_rows'):
                wc.max_table_size = int(data.get('max_table_rows'))
            self.config = wc
        elif isinstance(config, WeaverConfig):
            self.config = config
        else:
            self.config = WeaverConfig.from_env()

        # instantiate generators/executors
        self._gen = WeaverCodeGenerator(self.config)
        self._exec = WeaverCodeExecutor(self.config)

    def execute_query(self, table: pd.DataFrame, query: str, table_zoom: Optional[str] = None, table_desc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronously generate code and execute using Weaver components.

        Returns a dict compatible with the original TableZoomer expectations:
        { 'answer': ..., 'code': ..., 'plan': ..., 'execution_results': ... }
        """
        instr = {
            'query': query,
            'table_desc': table_desc or {},
            'table_zoom': table_zoom,
        }

        try:
            # generate code (async -> run synchronously)
            gen_out = __import__('asyncio').run(self._gen.run(json.dumps(instr)))
            # gen_out is a JSON string with {"prompt":..., "rsp": {...}}
            try:
                gen_obj = json.loads(gen_out)
            except Exception:
                # if generator returned raw plan/code string, wrap it
                gen_obj = {'prompt': instr, 'rsp': {'code': gen_out, 'plan': ''}}

            # execute generated code
            exec_in = json.dumps(gen_obj)
            exec_out = __import__('asyncio').run(self._exec.run(exec_in))
            try:
                exec_obj = json.loads(exec_out)
            except Exception:
                exec_obj = {'response': exec_out, 'code': gen_obj.get('rsp', {}).get('code', ''), 'plan': gen_obj.get('rsp', {}).get('plan', '')}

            # build result
            return {
                'answer': exec_obj.get('response', ''),
                'code': gen_obj.get('rsp', {}).get('code', ''),
                'plan': gen_obj.get('rsp', {}).get('plan', ''),
                'execution_results': exec_obj
            }

        except Exception as e:
            return {
                'answer': '',
                'code': '',
                'plan': '',
                'execution_results': {'error': str(e)}
            }