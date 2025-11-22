"""
Memory Management System for DietAI
Implements Phase 3: Memory Management System

This module handles:
- Message counter tracking
- Short-term memory summarization
- Long-term memory management
"""

import configparser
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class MemoryManager:
    """
    Manages short-term and long-term memory for the dietary assistant.
    
    Step 7: Message Counter System
    Step 8: Short-Term Memory Summarization
    Step 9: Long-Term Memory System
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize the Memory Manager.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Get paths from config
        base_path = Path(__file__).parent
        self.memory_dir = base_path / config.get('paths', 'memory', fallback='memory')
        self.short_term_file = self.memory_dir / 'short_term_memory.txt'
        self.long_term_file = self.memory_dir / 'long_term_memory.txt'
        
        # Get memory settings from config
        self.short_term_limit = config.getint('memory', 'short_term_limit', fallback=10)
        self.long_term_sessions = config.getint('memory', 'long_term_sessions', fallback=3)
        
        # Step 7: Initialize message counter
        self.message_counter = 0
        
        # Buffer to store raw messages before summarization
        self.message_buffer: List[Dict[str, str]] = []
        
        # Ensure memory directory exists
        self.memory_dir.mkdir(exist_ok=True)
        
        # Step 6: Ensure memory files exist
        self._ensure_memory_files()
    
    def _ensure_memory_files(self):
        """Step 6: Ensure memory files exist, create them if they don't."""
        if not self.short_term_file.exists():
            with open(self.short_term_file, 'w') as f:
                f.write("# Short-term memory for current session\n")
                f.write("# Stores up to 10 message pairs before summarization\n\n")
        
        if not self.long_term_file.exists():
            with open(self.long_term_file, 'w') as f:
                f.write("# Long-term memory for previous sessions\n")
                f.write("# Stores summaries of past conversations\n\n")
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the buffer and increment counter.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.message_buffer.append({"role": role, "content": content})
        
        # Step 7: Increment counter only for user messages
        if role == "user":
            self.message_counter += 1
    
    def should_summarize(self) -> bool:
        """
        Step 7: Check if summarization should be triggered.
        
        Returns:
            True if message counter reached threshold
        """
        return self.message_counter >= self.short_term_limit
    
    def summarize_short_term_memory(self, llm) -> str:
        """
        Step 8: Create Short-Term Memory Summarization.
        
        Collects the last 10 raw messages (5 user inputs, 5 assistant responses),
        sends to LLM for summarization, and appends to short_term_memory.txt.
        
        Args:
            llm: Initialized Llama model instance
            
        Returns:
            Summary text
        """
        if not self.message_buffer:
            return ""
        
        # Collect the last messages (up to the limit)
        messages_to_summarize = self.message_buffer[-self.short_term_limit:]
        
        # Format the conversation transcript
        transcript = "Conversation Transcript:\n"
        for msg in messages_to_summarize:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            transcript += f"{role_label}: {msg['content']}\n"
        
        # Create summarization prompt
        summarization_prompt = (
            "Summarize this conversation batch, focusing on meals discussed, "
            "any new preferences/allergies mentioned, and the recommended macro split.\n\n"
            f"{transcript}\n\n"
            "Provide a concise summary:"
        )
        
        # Get summary from LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes dietary conversations."},
            {"role": "user", "content": summarization_prompt}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            stream=False
        )
        
        summary = response['choices'][0]['message']['content']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append summary to short_term_memory.txt
        with open(self.short_term_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Summary ({timestamp}) ---\n")
            f.write(f"{summary}\n")
            f.write("-" * 50 + "\n")
        
        # Clear the message buffer and reset counter
        self.message_buffer = []
        self.message_counter = 0
        
        return summary
    
    def save_to_long_term_memory(self, llm):
        """
        Step 9: End of Conversation - Save to Long-Term Memory.
        
        Reads all content from short_term_memory.txt, sends to LLM for
        comprehensive summarization, appends to long_term_memory.txt,
        and clears short_term_memory.txt.
        
        Args:
            llm: Initialized Llama model instance
        """
        # Read all content from short_term_memory.txt
        if not self.short_term_file.exists():
            return
        
        with open(self.short_term_file, 'r', encoding='utf-8') as f:
            short_term_content = f.read()
        
        if not short_term_content.strip() or short_term_content.strip().startswith("#"):
            # No meaningful content to summarize
            return
        
        # Create comprehensive summarization prompt
        summarization_prompt = (
            "Consolidate this short-term history into key long-term dietary trends, "
            "macro compliance, and persistent recommendations.\n\n"
            f"{short_term_content}\n\n"
            "Provide a comprehensive summary focusing on:\n"
            "- Key dietary trends and patterns\n"
            "- Macro compliance over time\n"
            "- Persistent recommendations and preferences\n"
            "- Important health-related notes:"
        )
        
        # Get comprehensive summary from LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates comprehensive dietary summaries."},
            {"role": "user", "content": summarization_prompt}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
            stream=False
        )
        
        comprehensive_summary = response['choices'][0]['message']['content']
        session_date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append comprehensive summary to long_term_memory.txt
        with open(self.long_term_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Session Summary ({session_date}) - {timestamp} ===\n")
            f.write(f"{comprehensive_summary}\n")
            f.write("=" * 60 + "\n")
        
        # Clear short_term_memory.txt (keep header)
        with open(self.short_term_file, 'w', encoding='utf-8') as f:
            f.write("# Short-term memory for current session\n")
            f.write("# Stores up to 10 message pairs before summarization\n\n")
        
        # Clear message buffer and reset counter
        self.message_buffer = []
        self.message_counter = 0
    
    def load_long_term_memory_context(self) -> str:
        """
        Step 9: Start of Conversation - Load Long-Term Memory Context.
        
        Reads memory/long_term_memory.txt, extracts the last 2-3 session summaries,
        and formats them as context for the new conversation's system prompt.
        
        Returns:
            Formatted long-term memory context string
        """
        if not self.long_term_file.exists():
            return ""
        
        with open(self.long_term_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip() or content.strip().startswith("#"):
            return ""
        
        # Extract session summaries (sections between === markers)
        sessions = []
        current_session = []
        in_session = False
        
        for line in content.split('\n'):
            if line.startswith('===') and 'Session Summary' in line:
                if current_session and in_session:
                    sessions.append('\n'.join(current_session))
                current_session = [line]
                in_session = True
            elif in_session:
                current_session.append(line)
                if line.startswith('=' * 60):
                    sessions.append('\n'.join(current_session))
                    current_session = []
                    in_session = False
        
        # Get the last N sessions (from config)
        recent_sessions = sessions[-self.long_term_sessions:] if sessions else []
        
        if not recent_sessions:
            return ""
        
        # Format as context
        context = "Previous Session Summaries:\n"
        context += "-" * 60 + "\n"
        for session in recent_sessions:
            context += f"{session}\n\n"
        
        return context
    
    def get_short_term_memory_context(self) -> str:
        """
        Get the current short-term memory context.
        
        Returns:
            Formatted short-term memory context string
        """
        if not self.message_buffer:
            # Try to read from file if buffer is empty
            if self.short_term_file.exists():
                with open(self.short_term_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Return only summaries, not the header
                if content and not content.strip().startswith("#"):
                    return content
            return ""
        
        # Format current buffer
        context = "Current Session Messages:\n"
        for msg in self.message_buffer:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role_label}: {msg['content']}\n"
        
        return context
    
    def clear_short_term_memory(self):
        """
        Clear current session (short_term_memory.txt) but preserve long_term_memory.txt.
        Used by /clear command.
        """
        # Clear buffer and counter
        self.message_buffer = []
        self.message_counter = 0
        
        # Clear file (keep header)
        with open(self.short_term_file, 'w', encoding='utf-8') as f:
            f.write("# Short-term memory for current session\n")
            f.write("# Stores up to 10 message pairs before summarization\n\n")
    
    def get_long_term_history(self) -> str:
        """
        Get all summaries from long_term_memory.txt.
        Used by /history command.
        
        Returns:
            All long-term memory content
        """
        if not self.long_term_file.exists():
            return "No long-term memory found."
        
        with open(self.long_term_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip() or content.strip().startswith("#"):
            return "No previous session summaries found."
        
        return content

