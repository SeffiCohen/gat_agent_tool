#!/bin/bash
# Smoke test for the biomarker-discovery skill structure.
# Verifies: SKILL.md frontmatter parses, all reference + asset files non-empty,
# both scripts respond to --help, MCP config exists, all five new verifiers
# pass (initial_prompt, iteration_prompt, parse_initial_200,
# first_expression_baseline, plateau_default).
# Exit non-zero on any failure.

set -e
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$SKILL/../.." && pwd)"
PASS=0
FAIL=0

check() {
    local name="$1"
    local cond="$2"
    if [ "$cond" = "0" ]; then
        echo "[PASS] $name"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $name"
        FAIL=$((FAIL+1))
    fi
}

# 1. SKILL.md frontmatter parses as YAML
python3 -c "
import sys, yaml
text = open('$SKILL/SKILL.md').read()
parts = text.split('---', 2)
if len(parts) < 3: sys.exit('SKILL.md missing frontmatter delimiters')
fm = yaml.safe_load(parts[1])
assert 'name' in fm and 'description' in fm and 'version' in fm, f'frontmatter missing required keys: {list(fm.keys())}'
assert fm['name'] == 'biomarker-discovery', f'name mismatch: {fm[\"name\"]}'
" 2>/dev/null
check "SKILL.md frontmatter valid YAML with name/description/version" "$?"

# 2. Reference files exist and are non-empty
for f in references/diseases.md references/expression_dsl.md references/literature_review.md references/iteration_loop.md; do
    [ -s "$SKILL/$f" ]
    check "$f non-empty" "$?"
done

# 3. Asset files exist and are non-empty (template + 2 new prompt assets)
for f in assets/template_final_report.md assets/initial_prompt.md assets/iteration_prompt.md; do
    [ -s "$SKILL/$f" ]
    check "$f non-empty" "$?"
done

# 4. Scripts respond to --help
python3 "$SKILL/scripts/silent_real_auc.py" --help > /dev/null 2>&1
check "silent_real_auc.py --help works" "$?"

python3 "$SKILL/scripts/finalize.py" --help > /dev/null 2>&1
check "finalize.py --help works" "$?"

# 5. .mcp.json exists and parses as JSON with the expected server entry
python3 -c "
import json, sys
cfg = json.load(open('$REPO/.mcp.json'))
assert 'mcpServers' in cfg, 'missing mcpServers key'
assert 'gat-multi-scorer' in cfg['mcpServers'], 'missing gat-multi-scorer entry'
srv = cfg['mcpServers']['gat-multi-scorer']
assert srv['command'] == 'gat-agent-multi-mcp', f'wrong command: {srv[\"command\"]}'
" 2>/dev/null
check "repo .mcp.json valid with gat-multi-scorer" "$?"

# 6. Each oracle file pairs with its impl
for impl in scripts/silent_real_auc.py scripts/finalize.py; do
    base=$(basename "$impl" .py)
    [ -f "$SKILL/scripts/_${base}_oracle.py" ]
    check "oracle exists for $impl" "$?"
done

# 7. Each verifier script exists
for v in tests/verify_silent_real_auc.py \
         tests/verify_finalize.py \
         tests/verify_initial_prompt.py \
         tests/verify_iteration_prompt.py \
         tests/verify_parse_initial_200.py \
         tests/verify_first_expression_baseline.py \
         tests/verify_plateau_default.py \
         tests/verify_literature_review_subagents.py; do
    [ -s "$SKILL/$v" ]
    check "$v exists and non-empty" "$?"
done

# 8. Run the static-asset verifiers (fast, no GAT, no parquet eval).
python3 "$SKILL/tests/verify_initial_prompt.py" > /dev/null 2>&1
check "verify_initial_prompt.py exits 0" "$?"

python3 "$SKILL/tests/verify_iteration_prompt.py" > /dev/null 2>&1
check "verify_iteration_prompt.py exits 0" "$?"

python3 "$SKILL/tests/verify_plateau_default.py" > /dev/null 2>&1
check "verify_plateau_default.py exits 0" "$?"

python3 "$SKILL/tests/verify_literature_review_subagents.py" > /dev/null 2>&1
check "verify_literature_review_subagents.py exits 0" "$?"

# 9. Run validation-layer guard (parse_candidates over a 200-batch).
python3 "$SKILL/tests/verify_parse_initial_200.py" > /dev/null 2>&1
check "verify_parse_initial_200.py exits 0" "$?"

# 10. Run finalize.py picker test (synthetic CSV + stub parquet, no GAT).
python3 "$SKILL/tests/verify_first_expression_baseline.py" > /dev/null 2>&1
check "verify_first_expression_baseline.py exits 0" "$?"

echo ""
echo "$PASS PASS, $FAIL FAIL"
[ $FAIL -eq 0 ]
