#!/usr/bin/env python3
"""Standalone OpenAI function-calling example.

This is the same logic as `gat-agent-tool loop-openai` but inlined into a
single script so you can read it straight through, tweak the prompts, or copy
pieces into your own project. Requires:

    pip install 'gat-agent-tool[openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/openai_function_calling.py \\
        --checkpoint /path/to/gat_model.pt \\
        --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from gat_agent_tool import GatScorerTool


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_expression",
            "description": "Score a biomarker expression with the GAT. Returns predicted AUC in [0.5, 1.0] or null.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_expressions",
            "description": "Batch version of score_expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expressions": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                },
                "required": ["expressions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_list",
            "description": "Return legal features, operators, and rules. Call first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(name: str, args: dict, scorer: GatScorerTool) -> dict:
    if name == "score_expression":
        score = scorer.score(args["expression"])
        return {"expression": args["expression"], "score": score}
    if name == "score_expressions":
        exprs = list(args["expressions"])
        scores = scorer.score_batch(exprs)
        return {"results": [{"expression": e, "score": s} for e, s in zip(exprs, scores)]}
    if name == "get_feature_list":
        info = scorer.info()
        return {
            "features": scorer.feature_names,
            "operators": scorer.operators,
            "max_depth": info.max_depth,
            "rules": [
                "Features are case-sensitive; spell them exactly as listed.",
                "No numeric constants.",
                f"Depth <= {info.max_depth}.",
                "Operators: + - * /.",
            ],
        }
    return {"error": f"unknown tool {name}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--max_tool_calls", type=int, default=60)
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")

    from openai import OpenAI
    client = OpenAI()
    scorer = GatScorerTool(args.checkpoint)

    messages = [
        {"role": "system", "content": (
            "You design biomarker expressions. Use the GAT tools to score candidates "
            "and iterate until you find a high-AUC expression (target >= 0.75). "
            "Call get_feature_list first. Prefer score_expressions (batch) over "
            "individual calls. When done, list your top 3 expressions with scores."
        )},
        {"role": "user", "content": "Find the best expression you can within your budget."},
    ]
    tool_calls_used = 0

    for round_idx in range(args.max_rounds):
        resp = client.chat.completions.create(model=args.model, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message

        # Record assistant turn
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(entry)

        if not msg.tool_calls:
            print("--- Final assistant message ---")
            print(msg.content)
            return

        for tc in msg.tool_calls:
            if tool_calls_used >= args.max_tool_calls:
                result = {"error": "tool call budget exhausted"}
            else:
                try:
                    parsed = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                result = run_tool(tc.function.name, parsed, scorer)
                tool_calls_used += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })
            print(f"[round {round_idx}] {tc.function.name}({json.dumps(parsed)[:80]}…) -> "
                  f"{json.dumps(result)[:140]}…")

    print(f"Ran out of rounds ({args.max_rounds}); tool calls used: {tool_calls_used}.")


if __name__ == "__main__":
    main()
