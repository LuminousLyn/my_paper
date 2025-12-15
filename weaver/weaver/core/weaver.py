import os
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from .base import BaseQA, QAResult
from ..config.settings import WeaverConfig
from ..config.logging_config import get_logger
from ..database.manager import DatabaseManager
from ..llm.client import LLMClient
from ..data.preprocessor import TablePreprocessor
from ..prompts import load_prompt, configure_prompt_loader

logger = get_logger("core.weaver")


class TableQA(BaseQA):
    """Single‑table question answering system with **incremental result writing**."""

    # ----------------------------------------------------------------------------------
    # ▌Initialization ▐
    # ----------------------------------------------------------------------------------
    def __init__(self, config: Optional[WeaverConfig] = None) -> None:
        if config is None:
            config = WeaverConfig.from_env()
        super().__init__(config)

        # make prompt loader point to external dir if provided
        configure_prompt_loader(self.config.prompts_dir)

        self.data_loader = None  # lazy init if ever needed
        logger.info("TableQA initialised successfully (incremental mode)")

    # ----------------------------------------------------------------------------------
    # ▌Set‑ups ▐
    # ----------------------------------------------------------------------------------
    def _setup_database(self) -> None:
        conn_str = self.config.database.get_connection_string()
        db_type = self.config.database.db_type
        self.database = DatabaseManager(conn_str, db_type)
        logger.info(f"Database initialised: {db_type} – {conn_str}")

    def _setup_llm(self) -> None:
        self.llm = LLMClient(self.config.llm)
        logger.info(f"LLM client ready: {self.config.llm.model}")

    # ----------------------------------------------------------------------------------
    # ▌Public entrypoints ▐
    # ----------------------------------------------------------------------------------
    def ask(
        self,
        question_obj: Union[str, Dict[str, Any]],
        *, 
        include_token_stats: bool = False,
        **kwargs,
    ) -> QAResult:
        """Answer a single question (string or structured object)."""
        if isinstance(question_obj, str):
            # Simple string question - need table_path or table data
            question = question_obj
            table_path = kwargs.get('table_path')
            table_name = kwargs.get('table_name', 'table')
            
            if table_path:
                table = pd.read_csv(table_path)
            elif 'table' in kwargs:
                table = kwargs['table']
            else:
                raise ValueError("For string questions, provide table_path or table in kwargs")
            
            # Create a JSON object format
            question_obj = {
                'question': question,
                'table_name': table_name,
                'table': table,
                'paragraphs': kwargs.get('paragraphs'),
                'column_description_file': kwargs.get('column_description_file'),
                'table_schema_file': kwargs.get('table_schema_file'),
                'target_value': kwargs.get('target_value')  # for evaluation
            }
        
        return self._process_question(question_obj, include_token_stats, kwargs.get('dataset', 'default'))

    # ----------------------------------------------------------------------------------
    # ▌Dataset evaluation ▐
    # ----------------------------------------------------------------------------------
    def evaluate_dataset(
        self,
        dataset_name: str,
        data_path: str,
        *, 
        num_samples: Optional[int] = None,
        start_index: int = 0,
        include_token_stats: bool = False,
        flush_every: int = 20,
    ) -> List[QAResult]:
        """Evaluate on *dataset_name* saving results incrementally.

        Args:
            dataset_name: logical name (e.g. "wikitq").
            data_path: path to dataset json list.
            num_samples: truncate dataset (debug).
            start_index: offset to resume.
            include_token_stats: whether to attach token usage.
            flush_every: how many samples between file flushes.
        """
        logger.info(f"\n=== Evaluating {dataset_name} – source: {data_path} ===")

        # ------------------------------------------------------------------
        # Load dataset & decide range
        # ------------------------------------------------------------------
        with open(data_path, "r") as fh:
            data: List[Dict[str, Any]] = json.load(fh)

        end_index = len(data) if num_samples is None else min(start_index + num_samples, len(data))
        logger.info(f"Processing {end_index - start_index} samples (idx {start_index}..{end_index-1})")

        # ------------------------------------------------------------------
        # Prepare results file (append‑safe)
        # ------------------------------------------------------------------
        results_file = self.config.results_dir / (
            f"{self.config.llm.model.replace('/', '_')}_{dataset_name}_results.jsonl"
        )
        # ensure dir exists
        results_file.parent.mkdir(parents=True, exist_ok=True)
        # if we are resuming and file exists, read completed count to skip duplicates
        processed = 0
        if results_file.exists():
            with open(results_file, "r") as fh:
                processed = sum(1 for _ in fh)
            logger.info(f"Resuming – {processed} results already on disk")
        else:
            logger.info("Starting new results file: %s", results_file)

        # open file in append mode and keep handle across iterations
        results_fh = open(results_file, "a", buffering=1)  # line‑buffered

        results: List[QAResult] = []
        correct_count = 0

        for idx in range(start_index + processed, end_index):
            qobj = data[idx]
            logger.info("\n--- Sample %d / %d ---", idx + 1, end_index)
            res = self._process_question(qobj, include_token_stats, dataset_name)
            results.append(res)
            # update counters
            if res.is_correct:
                correct_count += 1

            # write single line as json – easy to stream & resume
            results_fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")

            # flush every *flush_every* samples
            if (idx + 1) % flush_every == 0:
                results_fh.flush()
                logger.info(
                    "[progress] %d/%d processed – running accuracy: %.2f%%",
                    idx + 1,
                    end_index,
                    100 * correct_count / len(results),
                )

        # final flush & close
        results_fh.flush()
        results_fh.close()
        logger.info("Evaluation finished – results stored at %s", results_file)
        return results
    
    # ----------------------------------------------------------------------------------
    # ▌Question Processing ▐
    # ----------------------------------------------------------------------------------
    def _process_question(self, question_obj: Dict[str, Any], include_token_stats: bool = False, dataset: str = "default") -> QAResult:
        """Process a single question object and return results."""
        # Save current dataset name for _load_table method
        self.current_dataset = dataset
        
        start_time = time.time()
        
        # Extract required fields
        question = question_obj['question']
        table_id = question_obj.get('table_id', 'unknown')
        table_name = question_obj.get('table_name', 'table')
        
        logger.info(f"Processing question: {question[:100]}...")
        logger.info(f"Table: {table_name} (ID: {table_id})")
        
        try:
            # Step 1: Load table data
            logger.info("Loading table data...")
            table = self._load_table(question_obj)
            logger.info(f"Table loaded successfully: {table.shape[0]} rows, {table.shape[1]} columns")
            
            # Step 2: Load optional context
            logger.info("Loading optional context (paragraphs, descriptions, schema)...")
            paragraphs = self._load_paragraphs(question_obj)
            column_descriptions = self._load_column_descriptions(question_obj) 
            table_schema = self._load_table_schema(question_obj)
            
            context_info = []
            if paragraphs:
                context_info.append(f"paragraphs ({len(paragraphs)} chars)")
            if column_descriptions:
                context_info.append("column descriptions")
            if table_schema:
                context_info.append("table schema")
            
            if context_info:
                logger.info(f"Context loaded: {', '.join(context_info)}")
            else:
                logger.info("No additional context provided")
            
            # Step 3: Preprocess table
            logger.info("Preprocessing table for SQL compatibility...")
            preprocessor = TablePreprocessor(
                max_column_width=self.config.max_table_size,
                max_rows=self.config.max_table_size
            )
            
            clean_table_name, clean_table = preprocessor.clean_table(table, table_name)
            logger.info(f"Table preprocessed: '{table_name}' → '{clean_table_name}'")

            # Step 4: Generate column descriptions if not provided
            if column_descriptions is None:
                logger.info("Generating column descriptions using LLM...")
                column_descriptions = self._generate_column_descriptions(clean_table, clean_table_name, question)
                logger.info("Column descriptions generated")
            else:
                logger.info("Using provided column descriptions")

            # Step 5: Filter relevant columns (if enabled)
            if self.config.filter_relevant_columns:
                logger.info("Filtering relevant columns using LLM...")
                original_cols = len(clean_table.columns)
                clean_table = preprocessor.filter_relevant_columns(
                    clean_table, question, column_descriptions, self.llm, paragraphs, clean_table_name
                )
                logger.info(f"Column filtering complete: {original_cols} → {len(clean_table.columns)} columns")
            else:
                logger.info("Column filtering disabled, using all columns")

            # Step 6: Upload table to database
            logger.info("Uploading table to database...")
            self.database.upload_table(clean_table_name, clean_table)
            logger.info(f"Table uploaded: {clean_table_name} ({len(clean_table)} rows, {len(clean_table.columns)} columns)")
            
            # Step 7: Extract relevant paragraphs if provided
            if paragraphs:
                logger.info("Extracting relevant information from paragraphs...")
                relevant_paragraphs = self._get_relevant_paragraphs(paragraphs, clean_table, question)
                logger.info("Relevant paragraphs extracted")
            else:
                relevant_paragraphs = "No additional information provided."
                logger.info("No paragraphs to process")
            
            # Step 8: Create execution plan
            logger.info("Creating execution plan using LLM...")
            plan = self._create_plan(clean_table, clean_table_name, question, column_descriptions, relevant_paragraphs, dataset)
            logger.info("Execution plan created")
            
            # Step 9: Verify plan
            logger.info("Verifying and improving plan...")
            verified_plan = self._verify_plan(plan, clean_table, clean_table_name, question, column_descriptions, relevant_paragraphs, dataset)
            logger.info("Plan verified and improved")

            # Step 10: Generate executable code
            logger.info("Generating executable code from plan...")
            code = self._generate_code(verified_plan, clean_table, clean_table_name, question, column_descriptions, relevant_paragraphs, dataset)
            logger.info("Code generated successfully")

            # Step 11: Execute the code
            logger.info("Executing generated code...")
            final_table = self._execute_code(code, clean_table_name, question, relevant_paragraphs)
            logger.info(f"Code execution complete: final table shape {final_table.shape}")
            logger.info("Final table preview:")
            logger.info(final_table.head())

            # Step 12: Extract final answer
            logger.info("Extracting final answer from result table...")
            answer = self._extract_answer(final_table, question, relevant_paragraphs, dataset)
            logger.info("Answer extracted successfully")
            
            execution_time = time.time() - start_time
            logger.info(f"Question processed in {execution_time:.2f}s")
            
            # Step 13: Format answer and check correctness if gold answer provided
            is_correct = None
            gold_answer = question_obj.get('target_value')
            if gold_answer is not None:
                logger.info("Formatting answer and checking correctness...")
                # Use proper answer formatting
                formatted_answer, is_correct = self._compare_answers(
                    final_table, question, gold_answer, relevant_paragraphs, dataset
                )
                answer = formatted_answer  # Use the formatted answer
                logger.info(f"Answer formatted and checked: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            else:
                logger.info("No gold answer provided, skipping correctness check")

            # Step 14: Get token statistics if requested
            token_stats = None
            if include_token_stats:
                logger.info("Collecting token usage statistics...")
                token_stats = self.llm.get_usage_stats()
                logger.info("Token statistics collected")

            logger.info(f"Question processing complete! Answer: {answer}")

            return QAResult(
                question=question,
                answer=answer,
                plan=verified_plan,
                sql_code=code,
                is_correct=is_correct,
                gold_answer=gold_answer,
                table_id=table_id,
                token_stats=token_stats
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error processing question after {execution_time:.2f}s: {e}")

            # Get token statistics if requested (even for errors)
            token_stats = None
            if include_token_stats:
                logger.info("Collecting token statistics for error case...")
                token_stats = self.llm.get_usage_stats()
                
            return QAResult(
                question=question,
                answer=f"Error: {str(e)}",
                table_id=table_id,
                token_stats=token_stats
            )
    
    # ----------------------------------------------------------------------------------
    # ▌Table Loading Methods ▐
    # ----------------------------------------------------------------------------------
    def _load_table(self, question_obj: Dict[str, Any]) -> pd.DataFrame:
        """加载表格数据并确保返回DataFrame类型"""
        # 如果表格数据直接在question_obj中提供
        if 'table' in question_obj:
            table_data = question_obj['table']
            # 检查是否已经是DataFrame
            if isinstance(table_data, pd.DataFrame):
                return table_data
            # 如果是字典列表，转换为DataFrame
            elif isinstance(table_data, list) and all(isinstance(item, dict) for item in table_data):
                return pd.DataFrame(table_data)
            # 如果是其他格式，尝试转换为DataFrame
            try:
                return pd.DataFrame(table_data)
            except:
                raise ValueError(f"Cannot convert provided table data to DataFrame: {type(table_data)}")
        
        # 如果提供了table_file_name或table_path
        table_path = question_obj.get('table_file_name') or question_obj.get('table_path')
        if not table_path:
            raise ValueError("No table data or path provided in question object")
        
        # 获取当前数据集
        dataset = getattr(self, 'current_dataset', 'default').lower()
        
        # 处理绝对路径vs相对路径
        if not os.path.isabs(table_path):
            # 为不同数据集添加正确的路径前缀
            # TabFact 数据集路径处理
            if dataset == 'tabfact' and not (table_path.startswith('TabFact/') or table_path.startswith('tabfact/')):
                table_path = f'TabFact/{table_path}'
            # WikiTQ 数据集路径处理
            elif dataset == 'wikitq' and not (table_path.startswith('WikiTQ/') or table_path.startswith('wikitq/')):
                table_path = f'WikiTQ/{table_path}'
            # FinQA 数据集路径处理 - 使用大写FINQA
            elif dataset == 'finqa' and not (table_path.startswith('FINQA/') or table_path.startswith('FinQA/') or table_path.startswith('finqa/')):
                table_path = f'FINQA/{table_path}'
            # OTT-QA 数据集路径处理
            elif dataset == 'ott-qa' or dataset == 'ottqa':
                if not table_path.startswith('OTT-QA/'):
                    table_path = f'OTT-QA/traindev_tables_tok/{table_path}'
        
        # 构建完整路径
        if not os.path.isabs(table_path):
            dataset_path = os.path.join(self.config.datasets_dir, table_path)
        else:
            dataset_path = table_path
        
        # 检查文件是否存在，如果不存在尝试添加扩展名
        if not os.path.exists(dataset_path):
            # 尝试添加常见的扩展名
            for ext in ['.csv', '.json']:
                ext_path = f"{dataset_path}{ext}"
                if os.path.exists(ext_path):
                    dataset_path = ext_path
                    break
            else:
                raise FileNotFoundError(f"Failed to load table from {table_path}: [Errno 2] No such file or directory")
        
        try:
            # 加载表格文件
            if dataset_path.endswith('.csv'):
                return pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                
                # 特殊处理 OTT-QA 数据集的 JSON 格式
                if dataset == 'ott-qa' or dataset == 'ottqa':
                    # OTT-QA 的 JSON 结构通常包含 headers 和 data 字段
                    if isinstance(table_data, dict) and 'headers' in table_data and 'data' in table_data:
                        headers = table_data['headers']
                        df_data = table_data['data']
                        # 确保所有行的长度一致
                        max_len = max(len(row) for row in df_data) if df_data else 0
                        df_data = [row + [''] * (max_len - len(row)) for row in df_data]
                        
                        return pd.DataFrame(df_data, columns=headers)
                
                # 普通 JSON 格式
                return pd.DataFrame(table_data)
            else:
                # 如果没有扩展名，尝试作为CSV和JSON都尝试一下
                try:
                    # 先尝试作为CSV
                    return pd.read_csv(dataset_path)
                except Exception as e1:
                    try:
                        # 再尝试作为JSON
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            table_data = json.load(f)
                        return pd.DataFrame(table_data)
                    except Exception as e2:
                        raise ValueError(f"Unsupported table format: {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load table from {dataset_path}: {e}")
            raise ValueError(f"Error loading table: {str(e)}")
        
        raise ValueError("No table data found in question object")
    
    def _load_paragraphs(self, question_obj: Dict[str, Any]) -> Optional[str]:
        """Load additional paragraphs if provided."""
        paragraphs = question_obj.get('paragraphs')
        if paragraphs and isinstance(paragraphs, str) and len(paragraphs.strip()) > 0:
            return paragraphs.strip()
        return None
    
    def _load_column_descriptions(self, question_obj: Dict[str, Any]) -> Optional[str]:
        """Load column descriptions from file if provided."""
        desc_file = question_obj.get('column_description_file')
        if desc_file and os.path.exists(desc_file):
            try:
                with open(desc_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to load column descriptions from {desc_file}: {e}")
        return None
    
    def _load_table_schema(self, question_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load table schema from file if provided."""
        schema_file = question_obj.get('table_schema_file')
        if schema_file and os.path.exists(schema_file):
            try:
                with open(schema_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load table schema from {schema_file}: {e}")
        return None
    
    # ----------------------------------------------------------------------------------
    # ▌LLM Processing Methods ▐
    # ----------------------------------------------------------------------------------
    def _generate_column_descriptions(self, table: pd.DataFrame, table_name: str, question: str) -> str:
        """Generate column descriptions using LLM."""
        prompt = f"""
        Give me the column name, data type, formatting needed in detail, and column descriptions in detail, for the context of question on the table.
        Also, give a small description of the table using table name and table data given.
        
        Table name: {table_name}
        Table columns: {list(table.columns)}
        Table preview:
        {table.head().to_html()}
        
        Question: {question}
        
        Provide detailed descriptions for each column and the overall table.
        """
        
        return self.llm.call(prompt)
    
    def _get_relevant_paragraphs(self, paragraphs: str, table: pd.DataFrame, question: str) -> str:
        """Extract relevant information from paragraphs using LLM."""
        if not paragraphs:
            return "No additional information provided."
        
        prompt = f"""
        Given the question, some paragraphs and the table, you need to extract the useful information in the paragraphs to answer the question.
        You can use the table columns context as well to extract the relevant information from the paragraphs.
        
        Paragraphs: {paragraphs}
        Table columns: {list(table.columns)}
        Question: {question}
        
        Extract and return only the relevant information from the paragraphs.
        """
        
        return self.llm.call(prompt)
    
    def _create_plan(self, table: pd.DataFrame, table_name: str, question: str, 
                    column_descriptions: str, relevant_paragraphs: str, dataset: str = "default") -> str:
        """Create execution plan using LLM."""
        # Load prompts using the new system
        plan_prompt = load_prompt("planner_prompt", dataset)
        few_shot = load_prompt("few_shot_plan", dataset)
        
        prompt = plan_prompt + "\n\n" + few_shot + f"""

        Solve for this:
        Table name: {table_name}
        {table.to_html()}
        Column descriptions: {column_descriptions}
        Paragraphs: {relevant_paragraphs}
        Question: {question}
        
        Only give the step by step plan and remove any extra explanation or Code.
        Output format:
        Step 1: SQL - [Instruction that can be used to write MySQL query]
        Step 2: Either SQL or LLM
        Step 3: ...
        
        Plan:
        """
        
        return self.llm.call(prompt)
    
    def _verify_plan(self, plan: str, table: pd.DataFrame, table_name: str, 
                    question: str, column_descriptions: str, relevant_paragraphs: str, dataset: str = "default") -> str:
        """Verify and improve the plan using LLM."""
        # Load prompts using the new system
        base_verify_prompt = load_prompt("verify_plan", dataset)
        
        # Construct full prompt with context
        verify_prompt = base_verify_prompt + f"""
        
        Table name: {table_name}
        Table: {table.to_html()}
        Column descriptions: {column_descriptions}
        Paragraphs: {relevant_paragraphs}
        Question: {question}
        
        Old Plan:
        {plan}
        """
        return self.llm.call(verify_prompt)

    def _generate_code(self, plan: str, table: pd.DataFrame, table_name: str,
                      question: str, column_descriptions: str, relevant_paragraphs: str, dataset: str = "default") -> str:
        """Generate executable code from plan."""
        # Load prompts using the new system
        execute_prompt = load_prompt("execute_prompt", dataset)
        
        prompt = execute_prompt + f"""
        
        Table name: {table_name}
        Paragraphs: {relevant_paragraphs}
        Schema: {list(table.columns)}
        Column Descriptions: {column_descriptions}
        
        Table: (This is a Sample table and the actual table can have more rows than below provided)
        {table.to_html()}
        
        Question: {question}
        Plan: {plan}
        
        Give me code for solving the question, and no other explanations. 
        Keep in mind the column data formats while writing SQL code.
        """
        
        return self.llm.call(prompt)
    
    def _execute_code(self, code: str, table_name: str, question: str, relevant_paragraphs: str) -> pd.DataFrame:
        """Execute the generated code and return final table."""
        df = self.database.get_table_data(table_name)
        tmp_df = df
        current_table_name = table_name
        
        # Split the code by steps and execute each one
        steps = re.split(r"Step \d+", code)
        logger.info('-----------------EXECUTING CODE------------------')
        
        for num, step in enumerate(steps):
            if not step.strip():
                continue
                
            try:
                if 'SQL' in step[:20] or 'sql' in step[:20]:
                    # Extract and execute SQL query
                    sql_pattern = r"\b(?:CREATE TABLE|SELECT)\b.*?;"
                    matches = re.findall(sql_pattern, step, re.DOTALL)
                    
                    for match in matches:
                        logger.info('--------------------SQL STEP--------------------------')
                        logger.info(match)
                        
                        result = self.database.execute_query(match)
                        
                        if result.success:
                            if result.data is not None:
                                # SELECT query returned data
                                tmp_df = result.data
                                logger.info(f"SQL returned {len(tmp_df)} rows")
                            elif result.table_name:
                                # CREATE TABLE query created new table
                                current_table_name = result.table_name
                                tmp_df = self.database.get_table_data(current_table_name)
                                logger.info(f"Created table: {current_table_name}")
                        else:
                            # SQL execution failed - suppress error log for cleaner output
                            logger.debug(f"SQL execution failed: {result.error}")
                            continue
                        
                        logger.debug(f"Current table:\n{tmp_df.to_string()}")

                elif 'LLM' in step[:20] or 'llm' in step[:20]:
                    logger.info('-----------------------LLM STEP---------------------------')
                    logger.info(step)
                    
                    # Extract information from LLM step
                    cols = self._get_prev_colname(step)
                    final_cols = []
                    for col in cols:
                        if col in tmp_df.columns:
                            final_cols.append(col)
                        else:
                            logger.warning(f"Column '{col}' not found in the DataFrame.")
                    
                    if not final_cols:
                        logger.warning("No valid columns found for LLM step, skipping")
                        continue

                    step_table_name = self._get_tablename(step)
                    if step_table_name:
                        current_table_name = step_table_name
                        tmp_df = self.database.get_table_data(current_table_name)
                    
                    step_prompt = self._get_new_prompt(step)
                    new_col_name = self._get_new_colname(step)
                    
                    if not step_prompt or not new_col_name:
                        logger.warning("Missing prompt or column name for LLM step, skipping")
                        continue

                    # Process in batches to handle large tables
                    batch_size = 10
                    new_col = []

                    for start in range(0, len(tmp_df), batch_size):
                        end = start + batch_size
                        batch_df = tmp_df.iloc[start:end]
                        batch_column_value = batch_df[final_cols]

                        # Create LLM prompt for this batch
                        llm_prompt = self._create_llmstep_prompt(
                            step_prompt, batch_column_value, relevant_paragraphs, question
                        )
                        
                        batch_col = self.llm.call(llm_prompt)
                        
                        # Split the response by '#' to get individual values
                        batch_values = batch_col.split('#')
                        
                        # Clean up any empty values
                        batch_values = [val.strip() for val in batch_values if val.strip()]
                        
                        # Handle length mismatch
                        if len(batch_values) != len(batch_column_value):
                            logger.warning(f"BATCH LENGTH MISMATCH: Expected {len(batch_column_value)}, got {len(batch_values)}")
                            
                            # Try to fix by padding or truncating
                            if len(batch_values) < len(batch_column_value):
                                # Pad with the last value or empty string
                                while len(batch_values) < len(batch_column_value):
                                    batch_values.append(batch_values[-1] if batch_values else "")
                            elif len(batch_values) > len(batch_column_value):
                                # Truncate to expected length
                                batch_values = batch_values[:len(batch_column_value)]
                        
                        new_col.extend(batch_values)
                    logger.info(f'Final new column values: {new_col}')
                    logger.info(f'Final column length: {len(new_col)}, Expected: {len(tmp_df)}')
                    
                    # Add new column if lengths match
                    if len(new_col) == len(tmp_df) and new_col_name:
                        tmp_df[new_col_name] = new_col
                        # Upload updated table back to database
                        self.database.upload_table(current_table_name, tmp_df)
                        logger.info(f'LLM updated table: {current_table_name}, with column: {new_col_name}')
                    else:
                        logger.error(f'LLM column length mismatch: expected {len(tmp_df)}, got {len(new_col)}')
                        logger.error(f"This will cause the step to fail!")
                    
                    logger.debug(f"Updated table:\n{tmp_df.to_string()}")
                    
                else:
                    logger.debug(f"Unrecognized step type in step {num}: {step[:50]}")

                # Check if table became empty
                if tmp_df.shape[0] == 0:
                    logger.warning("Table became empty after step, returning original table")
                    return df
                
                df = tmp_df

            except Exception as e:
                logger.error(f'Error in step {num}: {e}')
                continue

        logger.info(f'Final table shape: {df.shape}')
        return df
    
    def _extract_answer(self, final_table: pd.DataFrame, question: str, 
                       relevant_paragraphs: str, dataset: str) -> str:
        """Extract final answer from the result table."""
        # Load prompts using the new system
        prompt = load_prompt("extract_answer", dataset)
        
        prompt += f"""
        Table: {final_table.to_html(index=False)}
        Paragraphs: {relevant_paragraphs}
        Question: {question}
        Answer:
        """
        
        return self.llm.call(prompt)

    def _format_answer(self, final_table: pd.DataFrame, question: str, gold_answer: str,
                      relevant_paragraphs: str, dataset: str) -> str:
        """
        Format answer using dataset-specific prompts.
        """
        logger.info('-----------------FORMATTING ANSWER------------------')
        logger.debug(f'Relevant paragraphs: {relevant_paragraphs}')
        
        # First, extract answer from table using the existing method
        answer = self._extract_answer(final_table, question, relevant_paragraphs, dataset)
        logger.info(f"Initial extracted answer: {answer}")

        # Second step: Format the answer using dataset-specific formatting
        answer_formatting_prompt = load_prompt("format_answer", dataset)
        
        answer_formatting_prompt += f'''
        Solve for this-
        Answer: {answer}
        Gold Answer: {gold_answer}
        Your Output:
        '''
        
        # Get formatted answer
        formatted_answer = self.llm.call(answer_formatting_prompt)
        logger.info(f"Formatted answer: {formatted_answer}")
        return formatted_answer
    
    def _compare_answers(self, final_table: pd.DataFrame, question: str, gold_answer: str,
                        relevant_paragraphs: str, dataset: str) -> tuple:
        """
        Compare model answer with gold answer using proper formatting.
        """
        # Format the answer using dataset-specific prompts
        formatted_answer = self._format_answer(final_table, question, gold_answer, relevant_paragraphs, dataset)
        
        logger.info(f'Gold answer: {gold_answer}')
        logger.info(f'Model answer: {formatted_answer}')

        # Check if answers match
        if formatted_answer.strip() == str(gold_answer).strip():
            logger.info(f'Model answer: {formatted_answer} and Gold answer {gold_answer} match')
            is_correct = True
        else:
            logger.info(f'Model answer: {formatted_answer} and Gold answer {gold_answer} do not match')
            is_correct = False
        
        return formatted_answer, is_correct
    
    # Helper methods for code execution
    def _get_new_colname(self, step: str) -> Optional[str]:
        """Extract new column name from LLM step."""
        match = re.search(r"(?<=column name: )(\S+)", step, re.IGNORECASE)
        if match:
            new_column_name = match.group(1)
            new_column_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', new_column_name)
            logger.debug(f"New column name: {new_column_name}")
            return new_column_name
        else:
            logger.debug("New column name not found.")
            return None
            
    def _get_tablename(self, step: str) -> Optional[str]:
        """Extract table name from step."""
        match = re.search(r"(?<=table name: )(\S+)", step, re.IGNORECASE)
        if match:
            name = match.group(1)
            name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', name)
            logger.debug(f"Table name: {name}")
            return name
        else:
            logger.debug("Table name not found.")
            return None
            
    def _get_new_prompt(self, step: str) -> Optional[str]:
        """Extract LLM prompt from step."""
        match = re.search(r'(?<=LLM prompt: )(.*)', step, re.IGNORECASE)
        if match:
            llm_prompt = match.group(1)
            logger.debug(f"LLM Prompt: {llm_prompt}")
            return llm_prompt
        else:
            logger.debug("LLM prompt not found.")
            return None
            
    def _get_prev_colname(self, step: str) -> List[str]:
        """Extract previous column names from step."""
        match = re.search(r"(?<=to be used: )(.*)", step, re.IGNORECASE)
        if match:
            column_names = match.group(1)
            column_names = column_names.split(',')
            column_names = [col.strip() for col in column_names]
            return column_names
        else:
            logger.debug("Previous column names not found.")
            return []
            
    def _create_llmstep_prompt(self, llm_step: str, column_value: pd.DataFrame, 
                              paragraphs: str, question: str) -> str:
        """Create prompt for LLM step processing."""
        input_count = len(column_value)
        
        prompt = f"""
        Given a column and step you need to perform on it with some paragraphs which can be useful-
        
        Column: {column_value.to_string()}
        Step to solve the question: {llm_step}
        Question: {question}
        Paragraphs: {paragraphs}
        
        CRITICAL INSTRUCTIONS: 
        - You must provide EXACTLY {input_count} values in your response
        - Separate values by '#' character only
        - Do not provide any explanation or additional text
        - Return only a list (separate values by '#') that can be added to a dataframe as a new column
        - Any value should not be more than 3 words (or each value should be as short as possible)
        - Size of output column MUST be same as input column: {input_count} values
        - Example format: value1#value2#value3 (for 3 input rows)
        
        Your response (exactly {input_count} values separated by #):
        """
        return prompt