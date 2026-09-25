import sys
import io
from pathlib import Path

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

table_file = Path("results/tables/model_comparison.md")

if not table_file.exists():
    print(f"Error: {table_file} not found. Run 'python scripts/generate_tables.py' first.")
    sys.exit(1)

with open(table_file, "r", encoding="utf-8") as f:
    content = f.read()

print(content)
