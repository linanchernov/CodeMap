#!/usr/bin/env python3
"""
CodeMap Pro - AI-powered code understanding and documentation tool
Analyzes code files/directories and generates human-readable summaries using LLM
"""

import os
import sys
import argparse
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import ast
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import yaml  # pip install pyyaml

# Defer Anthropic import to LLMAnalyzer to avoid early failure

class EntityType(str, Enum):
    """Types of code entities"""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE = "module"


class EntityCategory(str, Enum):
    """Categories for code entities"""
    CORE = "core"           # Main business logic
    HELPER = "helper"       # Support functions
    INFRA = "infra"         # Setup/configuration
    UTILITY = "utility"     # General-purpose
    DATA = "data"          # Data processing
    API = "api"            # External calls
    UI = "ui"              # User interface
    DB = "db"              # Database operations
    SECURITY = "security"  # Auth/security
    TEST = "test"          # Testing code
    UNKNOWN = "unknown"


@dataclass
class CodeEntity:
    """Represents a function, class, or method in code"""
    entity_type: EntityType
    name: str
    file_path: str
    line_number: int
    signature: str
    source_snippet: str
    docstring: Optional[str] = None
    category: EntityCategory = EntityCategory.UNKNOWN
    description: Optional[str] = None
    complexity: int = 0  # Cyclomatic complexity
    parameters: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Other entities this depends on


@dataclass
class FileAnalysis:
    """Analysis result for a single file"""
    file_path: str
    entities: List[CodeEntity] = field(default_factory=list)
    total_lines: int = 0
    language: str = "unknown"
    file_hash: str = ""  # For caching


class CodeParser:
    """Parses code and extracts entities with complexity metrics"""

    LANG_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cpp': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.swift': 'swift',
        '.kt': 'kotlin',
    }

    @staticmethod
    def get_language(file_path: str) -> str:
        """Detect programming language by extension"""
        ext = Path(file_path).suffix.lower()
        return CodeParser.LANG_EXTENSIONS.get(ext, 'unknown')

    @staticmethod
    def calculate_complexity(node) -> int:
        """Calculate cyclomatic complexity for AST node"""
        complexity = 1  # Base complexity
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
            
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_BoolOp(self, node):
                # Each 'and'/'or' adds complexity
                self.complexity += len(node.values) - 1
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(node)
        return complexity + visitor.complexity

    @staticmethod
    def parse_python(file_path: str, content: str) -> List[CodeEntity]:
        """Parse Python file and extract entities with complexity"""
        entities = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return entities

        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                signature = f"class {node.name}:"
                if node.bases:
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        else:
                            bases.append(ast.unparse(base))
                    signature = f"class {node.name}({', '.join(bases)}):"

                snippet = CodeParser._get_snippet(lines, node.lineno, node.end_lineno)
                complexity = CodeParser.calculate_complexity(node)
                
                entity = CodeEntity(
                    entity_type=EntityType.CLASS,
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    signature=signature,
                    source_snippet=snippet,
                    docstring=docstring,
                    complexity=complexity,
                )
                entities.append(entity)

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_docstring = ast.get_docstring(item)
                        args = []
                        
                        # Process positional arguments
                        for arg in item.args.args:
                            args.append(arg.arg)
                        
                        # Process *args and **kwargs
                        if item.args.vararg:
                            args.append(f"*{item.args.vararg.arg}")
                        if item.args.kwarg:
                            args.append(f"**{item.args.kwarg.arg}")
                        
                        method_sig = f"def {item.name}({', '.join(args)})"
                        
                        # Add return annotation if present
                        if item.returns:
                            method_sig += f" -> {ast.unparse(item.returns)}"
                        method_sig += ":"
                        
                        method_snippet = CodeParser._get_snippet(
                            lines, item.lineno, item.end_lineno
                        )
                        method_complexity = CodeParser.calculate_complexity(item)
                        
                        entities.append(CodeEntity(
                            entity_type=EntityType.METHOD,
                            name=f"{node.name}.{item.name}",
                            file_path=file_path,
                            line_number=item.lineno,
                            signature=method_sig,
                            source_snippet=method_snippet,
                            docstring=method_docstring,
                            complexity=method_complexity,
                            parameters=args,
                        ))

            elif isinstance(node, ast.FunctionDef) and not any(
                isinstance(parent, ast.ClassDef) for parent in ast.iter_child_nodes(tree)
                if hasattr(parent, 'body') and node in parent.body
            ):
                # Top-level function
                docstring = ast.get_docstring(node)
                args = []
                
                for arg in node.args.args:
                    args.append(arg.arg)
                
                if node.args.vararg:
                    args.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")
                
                signature = f"def {node.name}({', '.join(args)})"
                if node.returns:
                    signature += f" -> {ast.unparse(node.returns)}"
                signature += ":"
                
                snippet = CodeParser._get_snippet(lines, node.lineno, node.end_lineno)
                complexity = CodeParser.calculate_complexity(node)
                
                entities.append(CodeEntity(
                    entity_type=EntityType.FUNCTION,
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    signature=signature,
                    source_snippet=snippet,
                    docstring=docstring,
                    complexity=complexity,
                    parameters=args,
                ))

        return entities

    @staticmethod
    def _get_snippet(lines: List[str], start: int, end: Optional[int], max_lines: int = 10) -> str:
        """Extract code snippet from source with bounds checking"""
        if start is None:
            return ""
        
        # Convert from 1-based to 0-based indexing
        start_idx = max(0, start - 1)
        
        # Handle None end line
        if end is None:
            end_idx = min(len(lines), start_idx + 5)  # Show next 5 lines
        else:
            end_idx = min(len(lines), end)
        
        # Adjust for inclusive end in ast vs exclusive in Python slicing
        if end_idx < len(lines):
            end_idx += 1
        
        snippet_lines = lines[start_idx:end_idx]
        
        # Trim if too long
        if len(snippet_lines) > max_lines:
            snippet_lines = snippet_lines[:max_lines] + ['    # ... (truncated)']
        
        # Add line numbers
        result = []
        for i, line in enumerate(snippet_lines):
            line_num = start_idx + i + 1
            result.append(f"{line_num:4d} | {line}")
        
        return '\n'.join(result)

    @staticmethod
    def parse_javascript(file_path: str, content: str) -> List[CodeEntity]:
        """Basic JavaScript/TypeScript parsing using regex"""
        entities = []
        lines = content.split('\n')
        
        # Regex patterns for JS/TS
        patterns = [
            # Class
            (r'^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)', EntityType.CLASS),
            # Function
            (r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)', EntityType.FUNCTION),
            # Arrow function (const x = ...)
            (r'^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(', EntityType.FUNCTION),
            # Method in class
            (r'^\s*(\w+)\s*\([^)]*\)\s*{', EntityType.METHOD),
        ]
        
        for i, line in enumerate(lines):
            line_num = i + 1
            for pattern, entity_type in patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    snippet = CodeParser._get_snippet(lines, line_num, line_num + 10)
                    
                    entities.append(CodeEntity(
                        entity_type=entity_type,
                        name=name,
                        file_path=file_path,
                        line_number=line_num,
                        signature=line.strip(),
                        source_snippet=snippet,
                    ))
                    break
        
        return entities

    @staticmethod
    def parse_file(file_path: str) -> FileAnalysis:
        """Parse a code file and return analysis"""
        analysis = FileAnalysis(file_path=file_path)

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
            return analysis

        # Calculate file hash for caching
        file_hash = hashlib.md5(content.encode()).hexdigest()
        analysis.file_hash = file_hash
        analysis.total_lines = len(content.split('\n'))
        analysis.language = CodeParser.get_language(file_path)

        # Parse based on language
        if file_path.endswith('.py'):
            analysis.entities = CodeParser.parse_python(file_path, content)
        elif any(file_path.endswith(ext) for ext in ['.js', '.ts', '.jsx', '.tsx']):
            analysis.entities = CodeParser.parse_javascript(file_path, content)
        else:
            # Basic generic parsing
            pass

        return analysis


class LLMAnalyzer:
    """Uses LLM to analyze code entities and generate descriptions"""

    SUPPORTED_PROVIDERS = ['anthropic', 'openai', 'gemini']

    def __init__(self, provider: str = 'anthropic', api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f'{provider.upper()}_API_KEY')
        
        if not self.api_key:
            raise ValueError(f"API key required for {provider}. Set {provider.upper()}_API_KEY env var or pass api_key.")
        
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model = "claude-3-5-sonnet-20241022"
            except ImportError:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        
        elif self.provider == 'openai':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.model = "gpt-4-turbo-preview"
            except ImportError:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        elif self.provider == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
                self.model = "gemini-pro"
            except ImportError:
                raise ImportError("Google Generative AI library not installed. Run: pip install google-generativeai")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def analyze_entities(self, entities: List[CodeEntity]) -> List[CodeEntity]:
        """Analyze code entities and add descriptions"""
        if not entities:
            return entities

        # Group by file for better context
        by_file = {}
        for entity in entities:
            if entity.file_path not in by_file:
                by_file[entity.file_path] = []
            by_file[entity.file_path].append(entity)

        print(f"\n🤖 Analyzing {len(entities)} code entities with {self.provider}...", file=sys.stderr)

        # Process in batches
        for file_path, file_entities in by_file.items():
            self._analyze_batch(file_entities)

        return entities

    def _analyze_batch(self, entities: List[CodeEntity], batch_size: int = 7):
        """Analyze a batch of entities from the same file"""
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}...", file=sys.stderr)
            self._process_batch(batch)

    def _process_batch(self, entities: List[CodeEntity]):
        """Send a batch to LLM for analysis"""
        prompt = self._build_prompt(entities)

        try:
            if self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    temperature=0.3,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                response_text = message.content[0].text
            
            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )
                response_text = response.choices[0].message.content
            
            elif self.provider == 'gemini':
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                response_text = response.text
            
            else:
                response_text = ""

            # Parse response and map back to entities
            self._parse_response(entities, response_text)

        except Exception as e:
            print(f"⚠️  LLM analysis failed: {e}", file=sys.stderr)
            # Fallback: use docstrings or generate basic descriptions
            for entity in entities:
                if entity.docstring:
                    entity.description = entity.docstring.split('\n')[0] + " [auto]"
                else:
                    entity.description = f"{entity.entity_type.value} '{entity.name}'"
                entity.category = EntityCategory.UNKNOWN

    def _build_prompt(self, entities: List[CodeEntity]) -> str:
        """Build a prompt for LLM to analyze entities"""
        entities_text = ""
        
        for i, entity in enumerate(entities, 1):
            complexity_info = f" (complexity: {entity.complexity})" if entity.complexity > 1 else ""
            
            entities_text += f"""
{'='*60}
Entity {i}:
Type: {entity.entity_type.value}{complexity_info}
Name: {entity.name}
Signature: {entity.signature}
Location: {entity.file_path}:{entity.line_number}
Docstring: {entity.docstring or "(no docstring)"}
Code snippet:
{entity.source_snippet}
{'='*60}
"""

        prompt = f"""Analyze these code entities and provide JSON output for each.

For each entity, analyze:
1. What this entity does (1-2 sentences, be specific)
2. Its category (core/helper/infra/utility/data/api/ui/db/security/test)
3. Key parameters and their purpose (if any)
4. Any potential issues or improvements

{entities_text}

Respond with a JSON array where each object has:
{{
  "name": "entity_name",
  "description": "clear, practical description",
  "category": "category_from_list",
  "parameters_analysis": ["param1: purpose", ...],
  "notes": "any observations or suggestions"
}}

Return ONLY valid JSON, no markdown, no explanations."""

        return prompt

    def _parse_response(self, entities: List[CodeEntity], response: str):
        """Parse LLM response and update entities"""
        try:
            # Extract JSON from response (handling possible extra text)
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                results = json.loads(json_str)
            else:
                # Try to parse line by line
                results = []
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            results.append(json.loads(line))
                        except:
                            continue
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}", file=sys.stderr)
            print(f"Response was:\n{response[:500]}...", file=sys.stderr)
            return

        # Map results back to entities by name
        result_by_name = {}
        for result in results:
            if 'name' in result:
                result_by_name[result['name']] = result
        
        for entity in entities:
            if entity.name in result_by_name:
                result = result_by_name[entity.name]
                entity.description = result.get('description', '')
                
                # Map category string to Enum
                category_str = result.get('category', '').lower()
                try:
                    entity.category = EntityCategory(category_str)
                except ValueError:
                    entity.category = EntityCategory.UNKNOWN
                
                # Store additional analysis
                if 'parameters_analysis' in result:
                    entity.parameters = result['parameters_analysis']
                if 'notes' in result:
                    entity.dependencies = [result['notes']]  # Reusing for now


class CodeCache:
    """Cache for analysis results to avoid redundant LLM calls"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), '.codemap_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, file_hash: str, provider: str) -> Path:
        """Get cache file path for given file hash and provider"""
        return self.cache_dir / f"{file_hash}_{provider}.json"
    
    def load(self, file_hash: str, provider: str) -> Optional[List[Dict]]:
        """Load cached analysis for file"""
        cache_file = self.get_cache_path(file_hash, provider)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def save(self, file_hash: str, provider: str, data: List[Dict]):
        """Save analysis to cache"""
        cache_file = self.get_cache_path(file_hash, provider)
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)


class CodeMapGenerator:
    """Generates various documentation formats from analysis"""

    @staticmethod
    def generate_markdown(analyses: List[FileAnalysis], title: str = "Code Map") -> str:
        """Generate markdown documentation with complexity metrics"""
        lines = [
            f"# {title}",
            f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Total analysis time: {datetime.now().strftime('%H:%M:%S')}*\n",
        ]

        # Summary stats
        total_files = len(analyses)
        total_entities = sum(len(a.entities) for a in analyses)
        avg_complexity = 0
        if total_entities > 0:
            all_complexities = [e.complexity for a in analyses for e in a.entities]
            avg_complexity = sum(all_complexities) / len(all_complexities)
        
        lines.append("## 📊 Summary")
        lines.append(f"- **Files analyzed:** {total_files}")
        lines.append(f"- **Code entities found:** {total_entities}")
        lines.append(f"- **Average complexity:** {avg_complexity:.1f}")
        
        # Category distribution
        categories = {}
        for analysis in analyses:
            for entity in analysis.entities:
                cat = entity.category.value
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            lines.append("\n### 📈 Category Distribution")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- **{cat}:** {count} entities")
        lines.append("")

        # High complexity warning
        high_complex_entities = []
        for analysis in analyses:
            for entity in analysis.entities:
                if entity.complexity >= 10:  # Threshold for high complexity
                    high_complex_entities.append(entity)
        
        if high_complex_entities:
            lines.append("### ⚠️ High Complexity Entities")
            lines.append("These entities have cyclomatic complexity >= 10 (consider refactoring):")
            for entity in high_complex_entities:
                lines.append(f"- `{entity.name}` (complexity: {entity.complexity}) in `{entity.file_path}`")
            lines.append("")

        # Organize by file
        for analysis in analyses:
            if not analysis.entities:
                continue

            rel_path = analysis.file_path
            lines.append(f"## 📄 {rel_path}")
            lines.append(f"*{analysis.language} | {analysis.total_lines} lines | Hash: `{analysis.file_hash[:8]}`*\n")

            # Group by type
            classes = [e for e in analysis.entities if e.entity_type == EntityType.CLASS]
            functions = [e for e in analysis.entities if e.entity_type == EntityType.FUNCTION]
            methods = [e for e in analysis.entities if e.entity_type == EntityType.METHOD]

            # Classes and methods
            if classes:
                lines.append("### 🏗️ Classes\n")
                for entity in classes:
                    lines.append(f"#### `{entity.name}`")
                    lines.append(f"*Line {entity.line_number} | Complexity: {entity.complexity} | Category: {entity.category.value}*")
                    
                    if entity.description:
                        lines.append(f"\n{entity.description}\n")
                    elif entity.docstring:
                        lines.append(f"\n{entity.docstring[:200]}...\n")
                    else:
                        lines.append(f"\n*(No description)*\n")

                    # List methods
                    class_methods = [m for m in methods if m.name.startswith(entity.name + '.')]
                    if class_methods:
                        lines.append("**Methods:**")
                        for method in class_methods:
                            method_name = method.name.split('.')[-1]
                            complexity_badge = f"🔴 " if method.complexity >= 7 else f"🟡 " if method.complexity >= 4 else f"🟢 "
                            lines.append(f"- {complexity_badge}`{method_name}`")
                            if method.description:
                                lines.append(f"  - {method.description}")
                            if method.parameters:
                                lines.append(f"  - Parameters: {', '.join(method.parameters)}")
                        lines.append("")

            # Standalone functions
            if functions:
                lines.append("### ⚡ Functions\n")
                for entity in functions:
                    complexity_badge = f"🔴 " if entity.complexity >= 7 else f"🟡 " if entity.complexity >= 4 else f"🟢 "
                    lines.append(f"- {complexity_badge}`{entity.signature}`")
                    lines.append(f"  *Line {entity.line_number} | Complexity: {entity.complexity} | Category: {entity.category.value}*")
                    if entity.description:
                        lines.append(f"  - {entity.description}")
                    if entity.parameters:
                        lines.append(f"  - Parameters: {', '.join(entity.parameters)}")
                lines.append("")

        # Add appendix with snippets for high complexity entities
        if high_complex_entities:
            lines.append("## 🔍 High Complexity Code Snippets")
            for entity in high_complex_entities[:5]:  # Limit to top 5
                lines.append(f"\n### `{entity.name}` (Complexity: {entity.complexity})")
                lines.append(f"```{analysis.language}")
                lines.append(entity.source_snippet[:500] + ("..." if len(entity.source_snippet) > 500 else ""))
                lines.append("```\n")

        return '\n'.join(lines)

    @staticmethod
    def generate_json(analyses: List[FileAnalysis]) -> str:
        """Generate JSON output for programmatic use"""
        output = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_files": len(analyses),
                "total_entities": sum(len(a.entities) for a in analyses),
            },
            "files": []
        }
        
        for analysis in analyses:
            file_info = {
                "path": analysis.file_path,
                "language": analysis.language,
                "lines": analysis.total_lines,
                "hash": analysis.file_hash,
                "entities": [asdict(entity) for entity in analysis.entities]
            }
            output["files"].append(file_info)
        
        return json.dumps(output, indent=2, default=str)

    @staticmethod
    def generate_html(analyses: List[FileAnalysis], title: str = "Code Map") -> str:
        """Generate HTML report"""
        # Basic HTML template - can be extended
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .entity {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .complexity-high {{ border-left: 5px solid #e74c3c; }}
        .complexity-medium {{ border-left: 5px solid #f39c12; }}
        .complexity-low {{ border-left: 5px solid #2ecc71; }}
        code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        for analysis in analyses:
            if analysis.entities:
                html += f'<h2>📄 {analysis.file_path}</h2>\n'
                html += f'<p><em>{analysis.language} | {analysis.total_lines} lines</em></p>\n'
                
                for entity in analysis.entities:
                    complexity_class = "complexity-high" if entity.complexity >= 7 else "complexity-medium" if entity.complexity >= 4 else "complexity-low"
                    html += f'<div class="entity {complexity_class}">\n'
                    html += f'<h3><code>{entity.name}</code></h3>\n'
                    html += f'<p><strong>Type:</strong> {entity.entity_type.value} | <strong>Line:</strong> {entity.line_number} | <strong>Complexity:</strong> {entity.complexity}</p>\n'
                    if entity.description:
                        html += f'<p>{entity.description}</p>\n'
                    html += f'<pre><code>{entity.signature}</code></pre>\n'
                    html += '</div>\n'
        
        html += "</body>\n</html>"
        return html


class CodeMap:
    """Main CodeMap utility with caching and multi-format output"""

    def __init__(self, 
                 provider: str = 'anthropic', 
                 api_key: Optional[str] = None,
                 use_cache: bool = True,
                 config_path: Optional[str] = None):
        
        self.provider = provider
        self.api_key = api_key
        self.use_cache = use_cache
        self.cache = CodeCache() if use_cache else None
        self.config = self._load_config(config_path)
        self.parser = CodeParser()
        
        try:
            self.llm = LLMAnalyzer(provider=provider, api_key=api_key)
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Tip: Install required package or use --provider openai/gemini", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            "exclude_patterns": [
                '**/__pycache__/**',
                '**/.venv/**',
                '**/venv/**',
                '**/.git/**',
                '**/node_modules/**',
                '**/.pytest_cache/**',
                '**/*.pyc',
                '**/*.pyo',
                '**/*.pyd',
                '**/*.so',
                '**/*.dll',
                '**/*.class',
            ],
            "include_extensions": list(CodeParser.LANG_EXTENSIONS.keys()),
            "max_file_size_mb": 10,
            "complexity_threshold_high": 10,
            "complexity_threshold_medium": 4,
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config {config_path}: {e}", file=sys.stderr)
        
        return default_config

    def analyze_path(self, path: str) -> List[FileAnalysis]:
        """Analyze a file or directory with caching"""
        path = Path(path)

        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

        files_to_analyze = self._collect_files(path)
        
        if not files_to_analyze:
            print(f"No code files found in {path}", file=sys.stderr)
            print(f"Supported extensions: {', '.join(self.config['include_extensions'])}", file=sys.stderr)
            sys.exit(1)

        print(f"🔍 Analyzing {len(files_to_analyze)} file(s)...", file=sys.stderr)

        analyses = []
        for file_path in sorted(files_to_analyze):
            print(f"  → {file_path}", file=sys.stderr)
            analysis = self.parser.parse_file(str(file_path))
            
            # Check cache for LLM analysis results
            if self.use_cache and self.cache and analysis.entities:
                cached = self.cache.load(analysis.file_hash, self.provider)
                if cached:
                    # Apply cached descriptions
                    for entity, cached_data in zip(analysis.entities, cached):
                        entity.description = cached_data.get('description')
                        entity.category = EntityCategory(cached_data.get('category', 'unknown'))
                    print(f"    (using cached analysis)", file=sys.stderr)
                else:
                    # Get new analysis from LLM and cache it
                    self.llm.analyze_entities(analysis.entities)
                    if self.cache:
                        cache_data = [
                            {
                                'name': entity.name,
                                'description': entity.description,
                                'category': entity.category.value
                            }
                            for entity in analysis.entities
                        ]
                        self.cache.save(analysis.file_hash, self.provider, cache_data)
            elif analysis.entities:
                # No cache, just analyze
                self.llm.analyze_entities(analysis.entities)
            
            analyses.append(analysis)

        return analyses

    def _collect_files(self, path: Path) -> List[Path]:
        """Collect files to analyze based on config"""
        files = []
        
        if path.is_file():
            if self._should_analyze_file(path):
                files.append(path)
        else:
            for ext in self.config['include_extensions']:
                for file_path in path.rglob(f'*{ext}'):
                    if self._should_analyze_file(file_path):
                        files.append(file_path)
        
        return files

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed based on config"""
        # Check exclude patterns
        for pattern in self.config['exclude_patterns']:
            if file_path.match(pattern):
                return False
        
        # Check file size
        max_size = self.config['max_file_size_mb'] * 1024 * 1024
        if file_path.stat().st_size > max_size:
            print(f"  Skipping {file_path} (>{self.config['max_file_size_mb']}MB)", file=sys.stderr)
            return False
        
        return True

    def print_summary(self, analyses: List[FileAnalysis]):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("CODE MAP PRO - ANALYSIS SUMMARY")
        print("=" * 80 + "\n")

        total_entities = sum(len(a.entities) for a in analyses)
        high_complex = sum(1 for a in analyses for e in a.entities if e.complexity >= 10)
        
        print(f"📊 Statistics:")
        print(f"   Files analyzed: {len(analyses)}")
        print(f"   Total entities: {total_entities}")
        print(f"   High complexity entities: {high_complex}")
        print()

        for analysis in analyses:
            if not analysis.entities:
                continue

            print(f"📄 {analysis.file_path}")
            print(f"   {analysis.language} | {analysis.total_lines} lines | {len(analysis.entities)} entities\n")

            for entity in analysis.entities[:5]:  # Show first 5 per file
                icon = "🏗️ " if entity.entity_type == EntityType.CLASS else "⚡ " if entity.entity_type == EntityType.FUNCTION else "🔧 "
                complexity_indicator = "🔴 " if entity.complexity >= 7 else "🟡 " if entity.complexity >= 4 else "🟢 "
                
                print(f"   {complexity_indicator}{icon}{entity.name}")
                if entity.description:
                    desc = entity.description[:100] + "..." if len(entity.description) > 100 else entity.description
                    print(f"      {desc}")
                print(f"      Type: {entity.entity_type.value} | Line: {entity.line_number} | Complexity: {entity.complexity}")
                print()

            if len(analysis.entities) > 5:
                print(f"   ... and {len(analysis.entities) - 5} more entities\n")

        print("=" * 80)
        print("💡 Tip: Use --md flag to generate detailed Markdown documentation")
        print("=" * 80)

    def save_output(self, analyses: List[FileAnalysis], 
                    format: str = 'markdown',
                    output_file: Optional[str] = None):
        """Save analysis in specified format"""
        if format == 'markdown':
            content = CodeMapGenerator.generate_markdown(analyses)
            default_file = "CODEMAP.md"
        elif format == 'json':
            content = CodeMapGenerator.generate_json(analyses)
            default_file = "codemap.json"
        elif format == 'html':
            content = CodeMapGenerator.generate_html(analyses)
            default_file = "codemap.html"
        else:
            raise ValueError(f"Unsupported format: {format}")

        output_path = Path(output_file or default_file)
        output_path.write_text(content, encoding='utf-8')
        print(f"\n✅ Saved {format.upper()} to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="CodeMap Pro - Understand and document code with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  codemap user_service.py
  codemap src/ --format markdown
  codemap . --provider openai --api-key $OPENAI_API_KEY
  codemap . --no-cache --format json --output analysis.json

Supported providers: anthropic (default), openai, gemini
"""
    )

    parser.add_argument(
        'path',
        help='File or directory to analyze'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['markdown', 'json', 'html'],
        default='markdown',
        help='Output format (default: markdown)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output file name (default depends on format)'
    )

    parser.add_argument(
        '--provider', '-p',
        choices=['anthropic', 'openai', 'gemini'],
        default='anthropic',
        help='LLM provider (default: anthropic)'
    )

    parser.add_argument(
        '--api-key',
        help='API key for LLM provider (overrides env var)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable analysis caching'
    )

    parser.add_argument(
        '--config',
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Initialize and run
    try:
        codemap = CodeMap(
            provider=args.provider,
            api_key=args.api_key,
            use_cache=not args.no_cache,
            config_path=args.config
        )

        analyses = codemap.analyze_path(args.path)
        
        if not args.verbose:
            codemap.print_summary(analyses)
        
        if args.format:
            codemap.save_output(analyses, format=args.format, output_file=args.output)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
