
# CodeMap Usage Examples

## Quick Start

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python codemap.py my_project.py
```

## Common Scenarios

### Scenario 1: Understand a single module

```bash
python codemap.py src/auth.py
```

Perfect for:
- Code review
- Before making changes
- Learning how authentication works

### Scenario 2: Onboard to a new project

```bash
python codemap.py src/ --md ONBOARDING.md
```

Then share `ONBOARDING.md` with new team members.

### Scenario 3: Document existing codebase

```bash
python codemap.py . \
  --exclude "tests/*,.venv/*,__pycache__/*" \
  --md docs/ARCHITECTURE.md
```

Commit to repo as living documentation.

### Scenario 4: Analyze multiple services

```bash
# Service 1
python codemap.py services/user_service/ --md docs/USER_SERVICE.md

# Service 2
python codemap.py services/payment_service/ --md docs/PAYMENT_SERVICE.md

# Create index
echo "# Services Documentation\n- [User Service](USER_SERVICE.md)\n- [Payment Service](PAYMENT_SERVICE.md)" > docs/SERVICES.md
```

### Scenario 5: Review before merging

```bash
# Generate map of feature branch
python codemap.py src/ --md FEATURE_MAP.md

# Review descriptions and structure before approving PR
git diff FEATURE_MAP.md
```

## CLI Cheat Sheet

```bash
# Basic analysis (output to console)
python codemap.py path/to/file.py

# Generate markdown
python codemap.py path/to/dir --md

# Custom output name
python codemap.py src/ --md --output CODEBASE.md

# Exclude patterns
python codemap.py . --exclude "tests/*,migrations/*" --md

# Full example
python codemap.py src/ \
  --exclude "tests/*,.venv/*,build/*" \
  --md docs/ARCHITECTURE.md \
  --api-key sk-ant-...
```

## Output Formats

### Console Output

Clean, emoji-based tree view with descriptions:

```text
📄 user_service.py
   (python | 156 lines)

  🔷 UserService
     → Main service handling user operations
     Category: core

  🔹 UserService.create_user(...)
     → Creates new user with validation
```

### Markdown Output

Structured, navigable documentation perfect for sharing:

```markdown
# Code Map

## Summary
- **Files analyzed:** 5
- **Code entities found:** 23

## src/user_service.py
*python | 156 lines*

### Classes

#### `UserService`
Main service handling user operations

- `create_user(...)`
  - Creates new user with validation
```

## Tips & Tricks

### Avoid analyzing dependencies

```bash
python codemap.py src/ --exclude "venv/*,site-packages/*" --md
```

### Focus on core logic

```bash
python codemap.py src/core --md CORE.md
```

### Generate per-module docs

```bash
for dir in src/*/; do
  module=$(basename "$dir")
  python codemap.py "$dir" --md "docs/$module.md"
done
```

### Watch for changes

```bash
# Generate initial map
python codemap.py . --md CODEBASE.md

# After changes
python codemap.py . --md CODEBASE.md
git diff CODEBASE.md
```

## Troubleshooting

### "anthropic library not installed"

```bash
pip install -r requirements.txt
```

### "No Python files found"

Check that:
1. Path exists: `ls your_path`
2. Contains Python files: `find your_path -name "*.py"`
3. Not all excluded: Review `--exclude` patterns

### "API key not found"

```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY="your-key"
python codemap.py .

# Option 2: Command line
python codemap.py . --api-key "your-key"
```

### Slow analysis

- Use `--exclude` to skip test/build directories
- Analyze smaller modules first
- Check Anthropic API status: https://status.anthropic.com

## Questions?

Open an issue or check existing examples!
