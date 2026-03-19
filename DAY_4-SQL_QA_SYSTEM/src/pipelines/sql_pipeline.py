"""
SQL QA Pipeline - FIXED VERSION
Better LLM + Fallback logic
"""

from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.sql_validator import SQLValidator
from src.utils.sql_executor import SQLExecutor
from src.utils.schema_loader import SchemaLoader


class SQLPipeline:
    """SQL Question Answering Pipeline"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        print("🚀 Initializing SQL QA Pipeline...")
        
        self.schema_loader = SchemaLoader(config_path)
        self.validator = SQLValidator(config_path)
        self.executor = SQLExecutor(config_path)
        
        # Try to load LLM generator
        self.use_llm = False
        try:
            from src.generator.sql_generator import SQLGenerator
            self.sql_generator = SQLGenerator(config_path)
            self.use_llm = True
        except Exception as e:
            print(f"⚠️  LLM not available: {e}")
        
        # Always have simple generator
        from src.generator.simple_sql_generator import SimpleSQLGenerator
        self.simple_generator = SimpleSQLGenerator(config_path)
        
        Path('outputs/queries').mkdir(parents=True, exist_ok=True)
        Path('outputs/results').mkdir(parents=True, exist_ok=True)
        
        print("✅ Pipeline ready!")
    
    def process_question(self, question: str) -> dict:
        """Process question through pipeline"""
        
        print(f"\n{'='*80}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*80}")
        
        result = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'sql': None,
            'validation': None,
            'execution': None,
            'summary': None,
            'success': False
        }
        
        sql = None
        source = None
        
        # STRATEGY 1: Try LLM first
        if self.use_llm:
            print("\n[1/4] 🤖 Generating SQL (LLM)...")
            gen_result = self.sql_generator.generate_with_retry(question)
            
            if gen_result['success']:
                sql = gen_result['sql']
                
                # Validate LLM output
                validation = self.validator.validate_all(sql)
                
                if validation['valid']:
                    # Try to execute
                    exec_result = self.executor.execute(sql, validate=False)
                    
                    if exec_result['success']:
                        source = "LLM"
                        print(f"   ✅ LLM SQL: {sql[:80]}...")
                    else:
                        print(f"   ⚠️  LLM SQL failed execution: {exec_result['error'][:50]}...")
                        sql = None
                else:
                    print(f"   ⚠️  LLM SQL failed validation: {validation['errors']}")
                    sql = None
            else:
                print(f"   ⚠️  LLM generation failed")
        
        # STRATEGY 2: Use simple generator as fallback
        if sql is None:
            print("\n[1/4] 🔧 Generating SQL (Simple)...")
            sql = self.simple_generator.generate_from_keywords(question)
            source = "Simple"
            print(f"   ✅ Simple SQL: {sql[:80]}...")
        
        result['sql'] = sql
        
        # Validate
        print("\n[2/4] ✅ Validating...")
        validation = self.validator.validate_all(sql)
        result['validation'] = validation
        
        if not validation['valid']:
            print(f"   ❌ Validation failed: {validation['errors']}")
            result['error'] = '; '.join(validation['errors'])
            return result
        
        print("   ✅ Valid")
        
        # Execute
        print("\n[3/4] ⚡ Executing...")
        execution = self.executor.execute(sql, validate=False)
        result['execution'] = {
            'success': execution['success'],
            'row_count': execution['row_count'],
            'error': execution['error']
        }
        
        if not execution['success']:
            print(f"   ❌ {execution['error']}")
            result['error'] = execution['error']
            return result
        
        print(f"   ✅ {execution['row_count']} rows (Source: {source})")
        
        # Summarize
        print("\n[4/4] 📊 Summarizing...")
        summary = self._summarize(question, sql, execution['data'])
        result['summary'] = summary
        result['success'] = True
        
        print(f"\n{'='*80}")
        print("📊 RESULTS")
        print(f"{'='*80}")
        print(summary)
        
        if execution['data'] is not None and len(execution['data']) > 0:
            print(f"\n📋 Data Preview:")
            print(execution['data'].head(10))
        
        self._save_results(result, execution['data'])
        
        return result
    
    def _summarize(self, question: str, sql: str, data: pd.DataFrame) -> str:
        """Generate summary"""
        
        if data is None or len(data) == 0:
            return "No results found"
        
        summary = f"Query returned {len(data)} row(s) with {len(data.columns)} column(s): {', '.join(data.columns)}"
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += "\n\nNumeric Summary:"
            for col in numeric_cols[:3]:  # Limit to 3 columns
                summary += f"\n  • {col}: min={data[col].min():.2f}, max={data[col].max():.2f}, avg={data[col].mean():.2f}"
        
        return summary
    
    def _save_results(self, result: dict, data: pd.DataFrame):
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_result = {
            'question': result['question'],
            'timestamp': result['timestamp'],
            'sql': result['sql'],
            'success': result['success']
        }
        
        query_file = Path(f'outputs/queries/query_{timestamp}.json')
        with open(query_file, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        if data is not None and len(data) > 0:
            csv_file = Path(f'outputs/results/results_{timestamp}.csv')
            data.to_csv(csv_file, index=False)
    
    def batch_process(self, questions: list) -> list:
        """Process multiple questions"""
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'#'*80}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'#'*80}")
            
            result = self.process_question(question)
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*80}")
        print(f"✅ Batch Complete: {successful}/{len(questions)} successful ({successful/len(questions)*100:.1f}%)")
        print(f"{'='*80}")
        
        return results


if __name__ == "__main__":
    pipeline = SQLPipeline()
    
    questions = [
        "How many artists?",
        "Show first 5 albums",
    ]
    
    pipeline.batch_process(questions)
