"""
@Desc: column linking.
@Author: xiongsishi
@Date: 2025-05-23.
"""

import json
from metagpt.actions import Action
import re
import ast

#for qwen3 output
def extract_from_content(s):
    parts = s.split("</think>")
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return s

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
        print("prompt:", prompt, "\n")
        original_prompt = prompt

        rsp = await self._aask(prompt)

        print("response:", rsp)
        rsp = rsp.strip()
        if rsp.startswith("```json") and rsp.endswith("```"):
            rsp = rsp.replace('```json', '').strip()
            rsp = rsp.replace('```', '')

        rsp = extract_from_content(rsp)
        
        # ---- query retrieval reflection ---
        # 移除了不必要的重复代码块，避免使用未定义变量和重复输出
        # ------

        return json.dumps({"prompt": original_prompt, "rsp": rsp}, ensure_ascii=False)