"""
SQL Generator - FIXED VERSION
Better SQL extraction to handle LLM verbosity
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
import yaml
from typing import Dict, Optional
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.schema_loader import SchemaLoader


class SQLGenerator:
    """Generate SQL queries from natural language using LLM"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        llm_config = self.config['llm']
        sql_config = self.config['sql_generation']
        
        print(f"🤖 Loading LLM: {llm_config['model_name']}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_config['model_name'])
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_config['model_name'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=llm_config['device']
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=llm_config['max_new_tokens'],
            temperature=llm_config['temperature'],
            do_sample=llm_config['do_sample']
        )
        
        # Load schema
        self.schema_loader = SchemaLoader(config_path)
        
        self.max_retries = sql_config['max_retries']
        self.include_schema = sql_config['include_schema']
        
        print("✅ SQL Generator ready!")
    
    def _build_prompt(self, question: str) -> str:
        """Build improved prompt"""
        
        tables = self.schema_loader.get_tables()
        
        schema_context = ""
        if self.include_schema:
            schema_context = "AVAILABLE TABLES:\n\n"
            
            for table in tables:
                columns = self.schema_loader.get_columns(table)
                col_names = [col['name'] for col in columns]
                
                schema_context += f"{table}: {', '.join(col_names)}\n"
        
        # Simpler, stricter prompt
        prompt = f"""Generate ONLY a SQL SELECT query. No explanations.

{schema_context}

Question: {question}

SQL:"""
        
        return prompt
    
    def generate(self, question: str) -> Dict:
        """Generate SQL query from question"""
        
        try:
            prompt = self._build_prompt(question)
            
            # Generate
            outputs = self.pipe(prompt)
            generated_text = outputs[0]['generated_text']
            
            # Extract ONLY the SQL
            sql = self._extract_clean_sql(generated_text, prompt)
            
            if not sql:
                return {
                    'success': False,
                    'sql': None,
                    'error': 'Failed to extract valid SQL'
                }
            
            return {
                'success': True,
                'sql': sql,
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'sql': None,
                'error': str(e)
            }
    
    def _extract_clean_sql(self, generated_text: str, prompt: str) -> Optional[str]:
        """Extract ONLY the SQL query, removing all explanations"""
        
        # Remove prompt
        if prompt in generated_text:
            sql_part = generated_text.split(prompt)[-1]
        else:
            sql_part = generated_text
        
        sql_part = sql_part.strip()
        
        # Method 1: Extract first complete SELECT statement
        # Look for SELECT ... ; and stop there
        select_match = re.search(r'(SELECT\s+.+?;)', sql_part, re.IGNORECASE | re.DOTALL)
        
        if select_match:
            sql = select_match.group(1)
            
            # Clean up: stop at first semicolon
            if ';' in sql:
                sql = sql.split(';')[0] + ';'
            
            # Remove any text after common separators
            for separator in ['\n\n', 'Explanation:', 'Example:', 'Output:', 'Question:', '  ']:
                if separator in sql:
                    sql = sql.split(separator)[0].strip()
                    if not sql.endswith(';'):
                        sql += ';'
            
            # Final cleanup
            sql = self._clean_sql(sql)
            
            return sql
        
        # Method 2: Manual extraction
        lines = sql_part.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Stop at explanations
            if any(keyword in line for keyword in ['Explanation:', 'Example:', 'Output:', 'Question:', 'Expected']):
                break
            
            # Start collecting from SELECT
            if 'SELECT' in line.upper() or len(sql_lines) > 0:
                sql_lines.append(line)
                
                # Stop at semicolon
                if ';' in line:
                    break
        
        if sql_lines:
            sql = ' '.join(sql_lines)
            sql = self._clean_sql(sql)
            
            if not sql.endswith(';'):
                sql += ';'
            
            return sql
        
        return None
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL query"""
        
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql)
        
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # Ensure single semicolon at end
        sql = sql.rstrip(';').strip() + ';'
        
        # Remove anything after semicolon
        if sql.count(';') > 1:
            sql = sql.split(';')[0] + ';'
        
        return sql.strip()
    
    def generate_with_retry(self, question: str) -> Dict:
        """Generate SQL with retry logic"""
        
        for attempt in range(self.max_retries):
            result = self.generate(question)
            
            if result['success']:
                # Validate the SQL doesn't contain garbage
                sql = result['sql']
                
                # Check for common issues
                if any(word in sql for word in ['Explanation', 'Example', 'Output', 'Expected']):
                    print(f"⚠️ Attempt {attempt + 1}: Extracted SQL contains explanations, retrying...")
                    continue
                
                result['attempt'] = attempt + 1
                return result
            
            print(f"⚠️ Attempt {attempt + 1} failed: {result['error']}")
        
        result['attempt'] = self.max_retries
        return result


if __name__ == "__main__":
    print("Testing SQL Generator...\n")
    
    generator = SQLGenerator()
    
    test_questions = [
        "Show all artists",
        "Count rows in artists",
        "Show first 5 rows from artists",
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = generator.generate(question)
        
        if result['success']:
            print(f"✅ SQL: {result['sql']}")
        else:
            print(f"❌ Error: {result['error']}")
