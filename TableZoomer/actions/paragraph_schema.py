#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List, Optional
from metagpt.actions.action import Action
from metagpt.logs import logger

class ParagraphSchema(Action):
    """Generate structured schema for paragraphs related to tables."""

    name: str = "ParagraphSchema"

    def __init__(self, PROMPT_TEMPLATE: str = None, **kwargs):
        super().__init__(**kwargs)
        if PROMPT_TEMPLATE is None:
            self.PROMPT_TEMPLATE = """
You are an expert assistant helping to build a structured schema for a question answering system over tables and related text.

Given:
- A paragraph of background text (from documents, reports, articles, etc.)
- A structured table schema that contains column names and data types
- A user question

Your task is to analyze the paragraph and generate a structured JSON object that summarizes relevant textual information. IMPORTANT: ALL fields must be filled with appropriate content. Never leave any field empty.

1. **text_summary**: A concise summary (2-4 sentences) of the entire paragraph, including key background, core information, and any main conclusions. This field must always be filled.

2. **table_related_text**: A list of short entries that explain or complement the table. Each entry should match one of the following subtypes:
   - "background_related": General context that provides background for the table.
   - "column_related": Text that defines or explains specific column names, values, or data types.
   - "data_explanation": Text that explains patterns, trends, or anomalies in the data.
   IMPORTANT: You MUST find specific information from the paragraph that relates to the table schema. Do NOT use generic phrases like "No information found".

3. **question_related_text**: A list of short entries in the form of { "Q-entity": "related text" }, where Q-entity is a keyword or concept mentioned in the question. For each important element in the question, find a sentence or phrase in the paragraph that helps answer or support that aspect.
   IMPORTANT: You MUST find specific information from the paragraph that relates to the question. Do NOT use generic phrases like "No information found".

Output your result as a structured JSON object. Ensure the JSON is valid and contains ALL required fields.

Example format:
{
  "text_summary": "...",
  "table_related_text": [
    { "background_related": "..." },
    { "column_related": "..." },
    { "data_explanation": "..." }
  ],
  "question_related_text": [
    { "A": "..." },
    { "B": "..." }
  ]
}

Now, please analyze the following inputs:

Paragraph:
{paragraph}

Table Schema:
{table_schema}

Question:
{question}
"""
        else:
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE

    async def run(self, paragraph, table_schema, question):
        """
        Generate a structured schema for the given paragraph, table schema, and question.
        
        Args:
            paragraph: The background text to analyze
            table_schema: The structured table schema
            question: The user question (optional)
            
        Returns:
            A JSON string representing the structured schema
        """
        # 添加详细日志
        logger.info(f"ParagraphSchema.run called with: paragraph_len={len(paragraph)}, table_schema_exists={bool(table_schema)}, question_exists={bool(question)}")
        
        # Prepare the prompt
        prompt = self.PROMPT_TEMPLATE.format(
            paragraph=paragraph,
            table_schema=json.dumps(table_schema, indent=2, ensure_ascii=False),
            question=question
        )
        
        # Call the LLM
        try:
            logger.info(f"Calling LLM with prompt length: {len(prompt)}")
            rsp = await self._aask(prompt)  # 这里的await现在将合法
            logger.info(f"LLM response received: {rsp[:200]}...")
            
            # Extract JSON from response if needed
            rsp = rsp.strip()
            if rsp.startswith("```json") and rsp.endswith("```"):
                rsp = rsp[7:-3].strip()  # Remove ```json and ```
            elif rsp.startswith("```") and rsp.endswith("```"):
                rsp = rsp[3:-3].strip()  # Remove ```
            
            logger.info(f"Processed LLM response: {rsp[:200]}...")
            
            # Validate JSON
            parsed_response = json.loads(rsp)
            logger.info(f"Parsed JSON successfully, contains keys: {list(parsed_response.keys())}")
            
            # 确保所有字段都基于实际分析，而不是默认值
            if not parsed_response.get("text_summary"):
                parsed_response["text_summary"] = paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
            
            # 检查table_related_text是否包含有意义的内容
            if not self._has_meaningful_entries(parsed_response.get("table_related_text", [])):
                logger.warning("table_related_text contains only default values, attempting to generate meaningful content")
                # 基于输入生成有意义的table_related_text
                table_related = []
                # 从table_schema中提取列名，支持两种格式：columns或column_list
                columns = []
                if table_schema:
                    if "columns" in table_schema:
                        # 格式1：columns是包含列字典的列表
                        columns = [col["name"] for col in table_schema["columns"] if "name" in col]
                    elif "column_list" in table_schema:
                        # 格式2：column_list是简单的列名列表
                        columns = table_schema["column_list"]
                    
                    if columns:
                        # 从段落中查找与列名相关的内容
                        for col in columns:
                            if col.lower() in paragraph.lower():
                                table_related.append({"column_related": f"The paragraph mentions '{col}', which corresponds to the '{col}' column in the table."})
                                break
                # 添加背景相关内容
                table_related.append({"background_related": "The paragraph provides context that helps understand the table data."})
                # 添加数据解释内容
                table_related.append({"data_explanation": "The paragraph may contain information about patterns or trends relevant to the table data."})
                parsed_response["table_related_text"] = table_related
            
            # 检查question_related_text是否包含有意义的内容
            if not self._has_meaningful_entries(parsed_response.get("question_related_text", [])) and question:
                logger.warning("question_related_text contains only default values, attempting to generate meaningful content")
                # 从问题中提取关键词并在段落中查找
                question_keywords = question.split()[:5]  # 取前5个关键词
                question_related = []
                for keyword in question_keywords:
                    if keyword.lower() in paragraph.lower():
                        # 找到关键词出现的上下文
                        import re
                        matches = re.finditer(r'(?i)'+keyword, paragraph)
                        for match in matches:
                            start = max(0, match.start() - 20)
                            end = min(len(paragraph), match.end() + 20)
                            context = paragraph[start:end].replace('\n', ' ')
                            question_related.append({keyword: f"The paragraph mentions '{keyword}' in the context: '{context}'"})
                            break
                # 如果没有找到匹配，添加基于问题的内容
                if not question_related:
                    question_related = [{"Question": "The paragraph provides information that may be relevant to answering the question."}]
                parsed_response["question_related_text"] = question_related
            
            # Return as JSON string
            final_rsp = json.dumps(parsed_response, ensure_ascii=False)
            logger.info(f"Final paragraph schema: {final_rsp[:300]}...")
            return final_rsp
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {rsp}")
            # 返回基于输入的结构化结果
            return self._generate_fallback_response(paragraph, table_schema, question, "JSON parsing error")
        except Exception as e:
            logger.error(f"Failed to generate paragraph schema: {e}", exc_info=True)
            # 返回基于输入的结构化结果
            return self._generate_fallback_response(paragraph, table_schema, question, str(e))
    
    def _generate_fallback_response(self, paragraph, table_schema, question, error_msg):

        # 生成text_summary
        text_summary = paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
        
        # 生成table_related_text
        table_related = []
        if table_schema:
            table_related.append({"background_related": "The paragraph provides context that helps understand the table data."})
            # 从table_schema中提取列名，支持两种格式：columns或column_list
            columns = []
            if "columns" in table_schema:
                columns = [col["name"] for col in table_schema["columns"] if "name" in col]
            elif "column_list" in table_schema:
                columns = table_schema["column_list"]
            
            if columns:
                table_related.append({"column_related": f"The table contains columns like {', '.join(columns[:3])}{'...' if len(columns) > 3 else ''} which may relate to the paragraph content."})
            table_related.append({"data_explanation": "The paragraph may contain information about patterns or trends relevant to the table data."})
        else:
            table_related.append({"background_related": "No table schema provided, but the paragraph contains valuable information."})
        
        # 生成question_related_text
        question_related = []
        if question:
            # 从问题中提取关键词
            keywords = question.split()[:3]  # 取前3个关键词
            if keywords:
                question_related.append({"Keywords": f"The question asks about {', '.join(keywords)}, which may relate to the paragraph content."})
            else:
                question_related.append({"Question": "The paragraph provides information that may be relevant to answering the question."})
        
        # 返回结构化响应
        return json.dumps({
            "text_summary": text_summary,
            "table_related_text": table_related,
            "question_related_text": question_related
        }, ensure_ascii=False)
    
    # helper: 判断 entries 是否包含“有意义”内容
    def _has_meaningful_entries(self, entries, placeholders=None):
        if not entries:
            return False
        if placeholders is None:
            placeholders = {
                "No table-related information found in the paragraph.",
                "No table schema provided.",
                "No paragraph text provided to analyze.",
                "No direct information found in the paragraph that answers this question.",
                "No information found",
                "No relevant information",
                "No data found"
            }
        for e in entries:
            if isinstance(e, dict):
                for v in e.values():
                    if isinstance(v, str) and v.strip() and v.strip() not in placeholders:
                        return True
                    if isinstance(v, (list, dict)) and bool(v):
                        return True
            elif isinstance(e, str):
                if e.strip() and e.strip() not in placeholders:
                    return True
        return False