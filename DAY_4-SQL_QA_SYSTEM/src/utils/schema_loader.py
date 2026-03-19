"""
Schema Loader
Loads and formats database schema for LLM context
"""

import sqlite3
import pandas as pd
from pathlib import Path
import yaml
from typing import List, Dict


class SchemaLoader:
    """Load and format database schema"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        db_config = config['database']
        
        if db_config['type'] == 'sqlite':
            self.db_path = db_config['sqlite_path']
            self.db_type = 'sqlite'
        else:
            raise NotImplementedError("PostgreSQL not yet supported")
        
        self.sql_config = config['sql_generation']
    
    def get_tables(self) -> List[str]:
        """Get list of all tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tables
    
    def get_columns(self, table_name: str) -> List[Dict]:
        """Get columns for a table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []
        
        for row in cursor.fetchall():
            columns.append({
                'name': row[1],
                'type': row[2],
                'notnull': row[3],
                'pk': row[5]
            })
        
        conn.close()
        return columns
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> pd.DataFrame:
        """Get sample data from table"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_schema_summary(self) -> str:
        """Get formatted schema summary"""
        tables = self.get_tables()
        
        summary = f"Database Schema ({len(tables)} tables):\n"
        summary += "=" * 80 + "\n\n"
        
        for table in tables:
            columns = self.get_columns(table)
            
            summary += f"📋 {table}\n"
            summary += f"   Columns ({len(columns)}):\n"
            
            for col in columns:
                pk = " (PRIMARY KEY)" if col['pk'] else ""
                summary += f"      • {col['name']} ({col['type']}){pk}\n"
            
            summary += "\n"
        
        return summary
    
    def get_schema_for_llm(self) -> str:
        """Get schema formatted for LLM prompt"""
        tables = self.get_tables()
        
        schema_text = "Database Schema:\n\n"
        
        for table in tables:
            columns = self.get_columns(table)
            
            # Table definition
            schema_text += f"Table: {table}\n"
            schema_text += "Columns:\n"
            
            for col in columns:
                schema_text += f"  - {col['name']} ({col['type']})"
                if col['pk']:
                    schema_text += " PRIMARY KEY"
                schema_text += "\n"
            
            # Sample data if enabled
            if self.sql_config.get('include_samples', True):
                try:
                    sample = self.get_sample_data(table, limit=self.sql_config.get('max_sample_rows', 3))
                    if len(sample) > 0:
                        schema_text += "\nSample Data:\n"
                        schema_text += sample.to_string(index=False)
                        schema_text += "\n"
                except:
                    pass
            
            schema_text += "\n"
        
        return schema_text


if __name__ == "__main__":
    print("Testing Schema Loader...\n")
    
    loader = SchemaLoader()
    
    print("📊 Tables:")
    for table in loader.get_tables():
        print(f"  • {table}")
    
    print("\n" + loader.get_schema_summary())
