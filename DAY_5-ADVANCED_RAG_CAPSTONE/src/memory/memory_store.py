"""
Memory Store - Conversational Memory Management
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Fix imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Try to load config
CONFIG_PATH = ROOT_DIR / 'config' / 'config.yaml'

class MemoryStore:
    """Manage conversational memory with last N messages"""
    
    def __init__(self, config_path: str = None):
        # Default configuration
        self.max_messages = 10
        self.store_path = ROOT_DIR / 'data' / 'chat_logs' / 'memory.json'
        self.memory_type = 'local'
        
        # Try to load from config
        if config_path is None:
            config_path = CONFIG_PATH
        
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            memory_config = config.get('memory', {})
            self.max_messages = memory_config.get('max_messages', 10)
            store_path = memory_config.get('store_path', 'data/chat_logs/memory.json')
            self.store_path = ROOT_DIR / store_path
            self.memory_type = memory_config.get('type', 'local')
        except Exception as e:
            print(f"⚠️  Using default config: {e}")
        
        # Create directory
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self._init_local()
        
        print(f"✅ Memory Store initialized (type: {self.memory_type}, max: {self.max_messages})")
    
    def _init_local(self):
        """Initialize local JSON storage"""
        if not self.store_path.exists():
            self._save_to_file({'conversations': {}})
    
    def _save_to_file(self, data: dict):
        """Save to JSON file"""
        with open(self.store_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_file(self) -> dict:
        """Load from JSON file"""
        if self.store_path.exists():
            try:
                with open(self.store_path) as f:
                    return json.load(f)
            except:
                return {'conversations': {}}
        return {'conversations': {}}
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ):
        """Add message to conversation history"""
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        data = self._load_from_file()
        
        if session_id not in data['conversations']:
            data['conversations'][session_id] = []
        
        data['conversations'][session_id].append(message)
        
        # Keep only last N messages
        data['conversations'][session_id] = \
            data['conversations'][session_id][-self.max_messages:]
        
        self._save_to_file(data)
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        
        data = self._load_from_file()
        history = data['conversations'].get(session_id, [])
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_context_window(self, session_id: str, n: int = 5) -> str:
        """Get last N messages as context string"""
        
        history = self.get_history(session_id, limit=n)
        
        if not history:
            return ""
        
        context = "Previous conversation:\n"
        
        for msg in history:
            role = msg['role'].capitalize()
            content = msg['content']
            context += f"{role}: {content}\n"
        
        return context
    
    def clear_session(self, session_id: str):
        """Clear conversation history for session"""
        
        data = self._load_from_file()
        if session_id in data['conversations']:
            del data['conversations'][session_id]
        self._save_to_file(data)
        
        print(f"✅ Session {session_id} cleared")
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        
        data = self._load_from_file()
        return list(data['conversations'].keys())
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        
        history = self.get_history(session_id)
        
        if not history:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'assistant_messages': 0,
                'first_message': None,
                'last_message': None
            }
        
        user_msgs = [m for m in history if m['role'] == 'user']
        assistant_msgs = [m for m in history if m['role'] == 'assistant']
        
        return {
            'total_messages': len(history),
            'user_messages': len(user_msgs),
            'assistant_messages': len(assistant_msgs),
            'first_message': history[0]['timestamp'] if history else None,
            'last_message': history[-1]['timestamp'] if history else None
        }


if __name__ == "__main__":
    # Test memory store
    print("Testing Memory Store...\n")
    
    memory = MemoryStore()
    
    session_id = "test_session_001"
    
    # Add messages
    memory.add_message(session_id, 'user', 'Hello!', {'intent': 'greeting'})
    memory.add_message(session_id, 'assistant', 'Hi! How can I help you?')
    
    # Get history
    print("📜 Conversation History:")
    history = memory.get_history(session_id)
    for msg in history:
        print(f"  {msg['role']}: {msg['content']}")
    
    # Stats
    print("\n📊 Session Stats:")
    stats = memory.get_session_stats(session_id)
    print(f"  Total messages: {stats['total_messages']}")
    
    print("\n✅ Memory Store test complete!")
