"""
LLM Actions for TableZoomer
"""
import json
from typing import Dict, Any, Optional
import pandas as pd

from metagpt.actions import Action, UserRequirement
from actions.query_analyse import extract_from_content
from actions.weaver_executor import TableZoomerExecutor
from weaver.config.settings import WeaverConfig

class LLMGenerate(Action):
    """基础LLM生成类"""
    name: str = "ThoughtGenerator"

    async def run(self, prompt: str):
        rsp = await self._aask(prompt)
        rsp = rsp.strip()
        if rsp.startswith("```json") and rsp.endswith("```"):
            rsp = rsp.replace('```json', '').strip()
            rsp = rsp.replace('```', '')
        return json.dumps(extract_from_content(rsp), ensure_ascii=False)

class TableLLMExecutor:
    """表格查询执行器类"""
    def __init__(self, config: Dict[str, Any]):
        """初始化LLM动作处理器
        
        Args:
            config: TableZoomer配置
        """
        self.config = config
        # 初始化Weaver执行器
        weaver_config = WeaverConfig.from_dict(config.get('weaver', {}))
        self.executor = TableZoomerExecutor(weaver_config)
        
    def execute_query(self,
                     table: pd.DataFrame,
                     query: str,
                     table_zoom: Optional[str] = None,
                     table_desc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用Weaver执行器执行查询
        
        Args:
            table: 输入表格
            query: 用户查询
            table_zoom: 压缩的表格信息(可选)
            table_desc: 表格描述(可选)
        
        Returns:
            执行结果字典
        """
        # 调用Weaver执行器执行查询
        result = self.executor.execute_query(
            table=table,
            query=query,
            table_zoom=table_zoom,
            table_desc=table_desc
        )
        
        return result