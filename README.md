# CodeMap
AI-powered code understanding utility Analyzes code files/directories and generates human-readable summaries using LLM
Quickly understand unfamiliar codebases by analyzing your Python code with Claude and generating human-readable summaries.

## Problem

You need to understand a new codebase fast:
- Onboarding to a new project
- Reviewing someone else's code
- Navigating a large unfamiliar module
- Documenting existing code

CodeMap **automates code comprehension** by:
1. Parsing Python files and extracting classes, methods, and functions
2. Sending code entities to Claude for analysis
3. Generating clear, practical summaries
4. Creating navigable markdown documentation

## Features

✨ **Fast Code Analysis** - Parses Python syntax and structure automatically  
🤖 **AI-Powered Summaries** - Uses Claude to generate practical descriptions  
📊 **Smart Categorization** - Identifies core logic, helpers, infrastructure, utilities  
📝 **Markdown Output** - Generate `CODEMAP.md` for easy navigation  
🚀 **CLI-First** - Simple, scriptable command-line interface  
🎯 **Flexible Scoping** - Analyze single files or entire directories  

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/codemap.git
cd codemap

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"
