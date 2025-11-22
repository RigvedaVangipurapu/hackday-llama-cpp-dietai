"""
DietAI - MVP Implementation
A personalized dietary assistant that remembers user preferences and gives recommendations.
Following instructions.txt (Steps 6-12, 17, 23) - MVP without RAG.
"""

import json
import os
import configparser
from pathlib import Path
from llama_cpp import Llama
from memory_manager import MemoryManager


class DietAI:
    """Main DietAI application class"""
    
    def __init__(self, config_path: str = "config.ini"):
        """Initialize DietAI with configuration"""
        print("🍎 Loading DietAI...")
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Paths
        self.user_data_path = Path(self.config['paths']['user_data'])
        self.preferences_file = self.user_data_path / "preferences.json"
        self.blood_report_file = self.user_data_path / "blood_report_summary.txt"
        
        # Load LLM model
        model_path = self.config['model']['model_path']
        print(f"📦 Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
            stream=True,
            max_tokens=200
        )
        
        # Load user data
        self.preferences = self._load_preferences()
        self.blood_report = self._load_blood_report()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(config_path)
        
        # Load long-term memory context
        self.long_term_context = self.memory_manager.load_long_term_memory_context()
        
        print("✅ DietAI ready!")
    
    def _load_preferences(self):
        """Load user preferences from JSON file"""
        try:
            with open(self.preferences_file, 'r') as f:
                prefs = json.load(f)
                print(f"📋 Loaded preferences: {prefs.get('dietary_style', 'Not specified')} diet")
                return prefs
        except FileNotFoundError:
            print("⚠️  No preferences file found")
            return {}
        except Exception as e:
            print(f"⚠️  Error loading preferences: {e}")
            return {}
    
    def _load_blood_report(self):
        """Load blood report summary"""
        try:
            with open(self.blood_report_file, 'r') as f:
                content = f.read()
                print("📊 Loaded blood report summary")
                return content
        except FileNotFoundError:
            print("⚠️  No blood report file found")
            return ""
        except Exception as e:
            print(f"⚠️  Error loading blood report: {e}")
            return ""
    
    def _build_system_prompt(self):
        """
        Step 12: Design the System Prompt
        Build comprehensive system prompt with user context
        """
        prompt = """You are an expert Personalized Nutrition and Macro Coach.

CRITICAL RULES:
- Keep ALL responses under 200 tokens
- Be PRECISE and CONCISE - maximum 2-3 sentences
- Always include macro breakdown (C/P/F) for meals
- Consider user's blood report and preferences
- Be direct and actionable

Note: You are not a registered dietitian or medical doctor."""
        
        return prompt
    
    def _build_personal_context(self):
        """Build personal context from preferences and blood report"""
        context = ""
        
        # Add preferences
        if self.preferences:
            context += "\n\nUSER PREFERENCES:\n"
            context += f"- Dietary Style: {self.preferences.get('dietary_style', 'Not specified')}\n"
            
            allergies = self.preferences.get('allergies', [])
            if allergies:
                context += f"- Allergies: {', '.join(allergies)} (NEVER recommend these foods)\n"
            
            cuisines = self.preferences.get('cuisine_preferences', [])
            if cuisines:
                context += f"- Preferred Cuisines: {', '.join(cuisines)}\n"
            
            macro_goals = self.preferences.get('macro_goals', {})
            if macro_goals:
                context += f"- Macro Goals: {macro_goals.get('carbohydrates_percent', 0)}% carbs, "
                context += f"{macro_goals.get('protein_percent', 0)}% protein, "
                context += f"{macro_goals.get('fat_percent', 0)}% fat\n"
        
        # Add blood report
        if self.blood_report:
            context += "\n\nBLOOD REPORT DATA:\n"
            context += self.blood_report
            context += "\n\nIMPORTANT: All recommendations must consider the blood report metrics above."
        
        return context
    
    def _build_complete_prompt(self, user_query: str):
        """
        Step 11: Construct Prompt
        Build complete prompt with all context in priority order
        """
        # System prompt (highest priority)
        system_prompt = self._build_system_prompt()
        
        # Personal context (preferences + blood report)
        personal_context = self._build_personal_context()
        
        # Long-term memory
        long_term_memory = self.long_term_context if self.long_term_context else ""
        
        # Short-term memory (recent conversation)
        short_term_memory = self.memory_manager.get_short_term_memory_context()
        
        # Build messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt + personal_context}
        ]
        
        # Add long-term memory if available
        if long_term_memory:
            messages.append({
                "role": "system", 
                "content": f"\n\nLONG-TERM MEMORY:\n{long_term_memory}"
            })
        
        # Add short-term memory if available
        if short_term_memory:
            messages.append({
                "role": "system",
                "content": f"\n\n{short_term_memory}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def _display_disclaimer(self):
        """
        Step 23: Add Dietary Disclaimer System
        Display disclaimer at startup
        """
        print("\n" + "="*80)
        print("⚠️  IMPORTANT DISCLAIMER")
        print("="*80)
        print("""
This dietary assistant is an informational tool and is NOT a substitute for:
- A licensed healthcare professional
- A registered dietitian
- Medical advice or treatment

The recommendations provided are general guidance and should not replace 
professional medical or nutritional advice. Always consult with qualified 
healthcare providers for:
- Chronic health conditions
- Severe dietary restrictions
- Medical concerns related to nutrition

If you experience severe symptoms or have concerns about your health, 
please seek immediate professional medical attention.
        """)
        print("="*80)
        print()
    
    def _check_emergency_keywords(self, user_input: str):
        """
        Step 23: Emergency Guardrails
        Check for emergency keywords and respond appropriately
        """
        emergency_keywords = [
            "anorexia", "suicide", "severe pain", "chest pain", 
            "heart attack", "stroke", "emergency", "dying"
        ]
        
        user_lower = user_input.lower()
        for keyword in emergency_keywords:
            if keyword in user_lower:
                print("\n" + "="*80)
                print("⚠️  EMERGENCY ALERT")
                print("="*80)
                print("""
This assistant cannot provide emergency medical assistance. 
If you are experiencing a medical emergency, please:
- Call 911 (or your local emergency number) immediately
- Contact your healthcare provider
- Go to the nearest emergency room

For mental health emergencies, contact:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
                """)
                print("="*80)
                return True
        return False
    
    def chat(self, user_message: str):
        """
        Step 11: Query Processing Pipeline
        Process user query and generate response
        """
        # Build complete prompt
        messages = self._build_complete_prompt(user_message)
        
        # Get LLM parameters from config
        temperature = float(self.config['llm']['temperature'])
        max_tokens = int(self.config['llm']['max_tokens'])
        repeat_penalty = float(self.config['llm']['repeat_penalty'])
        
        # Generate response with streaming enabled
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            stream=True
        )
        
        # Collect and display streaming response
        assistant_message = ""
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                token = delta['content']
                print(token, end="", flush=True)
                assistant_message += token
        
        print()  # New line after streaming completes
        
        # Update memory (Step 11: Update Memory Buffer)
        self.memory_manager.add_message("user", user_message)
        self.memory_manager.add_message("assistant", assistant_message)
        
        # Check if summarization is needed (Step 8)
        if self.memory_manager.should_summarize():
            print("\n[System: Summarizing conversation batch...]")
            self.memory_manager.summarize_short_term_memory(self.llm)
        
        return assistant_message
    
    def run(self):
        """
        Step 10: Build the Main Conversation Loop
        Main interactive loop with command handling
        """
        # Display disclaimer
        self._display_disclaimer()
        
        # Welcome message
        print("\n" + "="*80)
        print("🍎 Welcome to DietAI - Your Personal Dietary Assistant")
        print("="*80)
        
        # Display user profile
        if self.preferences:
            print("\n📋 Your Profile:")
            print(f"   Diet: {self.preferences.get('dietary_style', 'Not set')}")
            if self.preferences.get('allergies'):
                print(f"   Allergies: {', '.join(self.preferences.get('allergies', []))}")
            print()
        
        print("💬 Ask me anything about diet, nutrition, meal planning, or recipes!")
        print("\n📝 Commands:")
        print("   /exit    - End conversation and save session")
        print("   /clear   - Clear current session memory")
        print("   /history - View conversation history")
        print("   /goals   - View your preferences and blood report")
        print()
        
        # Main conversation loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Step 17: Handle special commands
                if user_input.startswith('/'):
                    if user_input.lower() == '/exit':
                        print("\n[System: Saving session...]")
                        self.memory_manager.save_to_long_term_memory(self.llm)
                        print("\n👋 Thanks for using DietAI! Stay healthy!")
                        break
                    
                    elif user_input.lower() == '/clear':
                        self.memory_manager.clear_short_term_memory()
                        print("✅ Current session cleared. Starting fresh!\n")
                        continue
                    
                    elif user_input.lower() == '/history':
                        print("\n" + "="*80)
                        print("CONVERSATION HISTORY")
                        print("="*80)
                        history = self.memory_manager.get_long_term_history()
                        print(history)
                        print("="*80 + "\n")
                        continue
                    
                    elif user_input.lower() == '/goals':
                        print("\n" + "="*80)
                        print("YOUR DIETARY PROFILE")
                        print("="*80)
                        if self.preferences:
                            print("\nPreferences:")
                            print(json.dumps(self.preferences, indent=2))
                        if self.blood_report:
                            print("\nBlood Report Summary:")
                            print(self.blood_report)
                        print("="*80 + "\n")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {user_input}")
                        print("Available commands: /exit, /clear, /history, /goals\n")
                        continue
                
                # Check for emergency keywords
                if self._check_emergency_keywords(user_input):
                    continue
                
                # Process normal query
                print("\nDietAI: ", end="", flush=True)
                response = self.chat(user_input)
                print()
                
            except KeyboardInterrupt:
                print("\n\n[System: Saving session...]")
                self.memory_manager.save_to_long_term_memory(self.llm)
                print("👋 Thanks for using DietAI! Stay healthy!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print("Please try again.\n")


if __name__ == "__main__":
    try:
        diet_ai = DietAI()
        diet_ai.run()
    except Exception as e:
        print(f"❌ Failed to start DietAI: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure:")
        print("1. The model file exists at the path specified in config.ini")
        print("2. You have installed requirements: pip install -r requirements.txt")
        print("3. config.ini is properly configured")
