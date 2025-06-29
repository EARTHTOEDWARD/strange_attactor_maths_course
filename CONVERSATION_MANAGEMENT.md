# Conversation Management Protocol

## Auto-Summary Instructions for GitHub Copilot

### For the Human:
1. **Count interactions** in the current chat session
2. **Every 10 prompts**, ask: "Can you summarize what we've accomplished?"
3. **After summary**, consider starting a fresh chat with the summary as context

### For GitHub Copilot:
When asked to summarize, provide:

1. **Session Overview**
   - Number of interactions
   - Time spent
   - Main objectives

2. **Technical Accomplishments**
   - Files created/modified
   - Functions implemented
   - Bugs fixed
   - Features added

3. **Current Status**
   - What's working
   - What's in progress
   - Next steps

4. **Context for Fresh Chat**
   - Key decisions made
   - Important constraints
   - Project state

### Template Response:
```
## Session Summary (X interactions)

**Accomplished:**
- [List key achievements]

**Files Modified:**
- [List files with brief description]

**Current State:**
- [Project status]

**For Fresh Chat:**
- [Essential context to carry forward]

**Next Steps:**
- [Recommended actions]
```

### Automation Triggers:
- VS Code extension could potentially track this
- GitHub Copilot Labs features might support this
- Custom workspace commands could implement counting

---
*Save this file as a reference for managing long conversations*
