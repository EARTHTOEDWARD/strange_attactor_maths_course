#!/usr/bin/env python3
"""
Conversation Tracker for GitHub Copilot Sessions
Helps manage long conversations by tracking interactions and suggesting summaries.
"""

import json
import os
from datetime import datetime
from pathlib import Path

class ConversationTracker:
    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.tracker_file = Path(self.workspace_path) / '.conversation-tracker.json'
        self.data = self.load_data()
        
    def load_data(self) -> dict:
        """Load existing conversation data or create new."""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {
            'sessions': [],
            'current_session': {
                'start_time': datetime.now().isoformat(),
                'interactions': 0,
                'topics': [],
                'files_modified': [],
                'summary': ''
            }
        }
    
    def save_data(self):
        """Save conversation data to file."""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def log_interaction(self, topic: str = None, files_modified: list = None):
        """Log a new interaction."""
        self.data['current_session']['interactions'] += 1
        
        if topic:
            self.data['current_session']['topics'].append(topic)
        
        if files_modified:
            self.data['current_session']['files_modified'].extend(files_modified)
        
        self.save_data()
        
        # Check if we should suggest a summary
        if self.data['current_session']['interactions'] % 10 == 0:
            return self.suggest_summary()
        
        return None
    
    def suggest_summary(self) -> str:
        """Generate a summary suggestion message."""
        interactions = self.data['current_session']['interactions']
        topics = list(set(self.data['current_session']['topics']))
        files = list(set(self.data['current_session']['files_modified']))
        
        message = f"""
🤖 **Conversation Checkpoint** ({interactions} interactions)

**Topics covered:** {', '.join(topics) if topics else 'Various'}
**Files modified:** {len(files)} files
**Session duration:** {self._get_session_duration()}

Would you like me to:
1. Summarize what we've accomplished?
2. Start a fresh chat to keep context manageable?

Type 'summarize' for option 1, or start a new chat for option 2.
"""
        return message
    
    def generate_summary(self) -> str:
        """Generate a detailed summary of the current session."""
        session = self.data['current_session']
        
        summary = f"""
# Session Summary ({session['interactions']} interactions)

**Started:** {session['start_time']}
**Duration:** {self._get_session_duration()}

## Topics Covered
{chr(10).join(f"• {topic}" for topic in set(session['topics']))}

## Files Modified ({len(set(session['files_modified']))})
{chr(10).join(f"• {file}" for file in set(session['files_modified']))}

## Key Accomplishments
{session.get('summary', 'No detailed summary recorded')}

---
*This summary was auto-generated. Consider starting a fresh chat for optimal context management.*
"""
        
        # Archive this session
        self.data['sessions'].append(session.copy())
        self.data['current_session'] = {
            'start_time': datetime.now().isoformat(),
            'interactions': 0,
            'topics': [],
            'files_modified': [],
            'summary': ''
        }
        self.save_data()
        
        return summary
    
    def _get_session_duration(self) -> str:
        """Calculate session duration."""
        start = datetime.fromisoformat(self.data['current_session']['start_time'])
        duration = datetime.now() - start
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m"

# Usage example
if __name__ == "__main__":
    tracker = ConversationTracker()
    
    # Simulate logging an interaction
    suggestion = tracker.log_interaction(
        topic="README.md updates", 
        files_modified=["README.md"]
    )
    
    if suggestion:
        print(suggestion)
    else:
        print(f"Interaction logged. Total: {tracker.data['current_session']['interactions']}")
