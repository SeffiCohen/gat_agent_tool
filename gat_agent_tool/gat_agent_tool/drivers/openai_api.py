"""gat_agent_tool.drivers.openai_api — iterative loop with an OpenAI chat model.

This is the **agentic** path: the OpenAI model is given the GAT as a tool via
native function-calling, and it decides when to score a candidate, when to
propose another batch, and when to stop. Compared to the local-HF driver
(which injects scored history into the prompt text), this path lets the model
interleave reasoning, scoring, and refinement in its own way.

Three tools are exposed to the model:
  * `score_expression(expression)`
  * `score_expressions(expressions)`
  * `get_feature_list()`

The loop is bounded by ``--max_tool_calls`` (hard cap on total GAT calls) and
``--max_rounds`` (hard cap on assistant turns). Every tool call, score, and
message is logged to the output directory so you can replay the session.

Requires an OpenAI API key: set ``OPENAI_API_KEY`` in the environment.

Example:
    export OPENAI_API_KEY=sk-...
    gat-agent-tool loop-openai \\
        --checkpoint /path/to/gat_model.pt \\
        --model gpt-4o-mini \\
        --output_dir out/ \\
        --max_rounds 30 --max_tool_calls 150 \\
        --topic "biomarker expressions for rheumatoid arthritis"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core import GatScorerTool  # noqa: F401


logger = logging.getLogger(__name__)


_DEFAULT_TOPIC = "biomarker expressions for clinical prediction from routine lab features"


def _tool_specs() -> List[Dict[str, Any]]:
    """JSON Schema tool specs for OpenAI's Chat Completions API."""
    return [
        {
            "type": "function",
            "function": {
                "name": "score_expression",
                "description": (
                    "Score a single biomarker expression with the trained GAT. "
                    "Returns a predicted AUC in [0.5, 1.0], or a null score with "
                    "an error message if the expression cannot be parsed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The expression to score, e.g. '(lab_X + lab_Y) / lab_Z'.",
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "score_expressions",
                "description": (
                    "Score many expressions in a single call (batched through the GAT). "
                    "Prefer this over repeated score_expression calls when you have a batch of candidates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expressions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of candidate expression strings.",
                            "minItems": 1,
                        }
                    },
                    "required": ["expressions"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_feature_list",
                "description": (
                    "Return the features, operators, and rules the GAT was trained on. "
                    "Call this FIRST — expression names must match exactly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Execute one tool call against the local GatScorerTool and return the
# JSON-serialisable result the model will see.
# ---------------------------------------------------------------------------
def _run_tool(name: str, arguments: Dict[str, Any], scorer: "GatScorerTool") -> Dict[str, Any]:
    try:
        if name == "score_expression":
            expr = str(arguments["expression"])
            score = scorer.score(expr)
            return {
                "expression": expr,
                "score": score,
                "error": None if score is not None else "could not parse expression",
            }
        if name == "score_expressions":
            exprs = [str(e) for e in arguments["expressions"]]
            scores = scorer.score_batch(exprs)
            return {
                "results": [
                    {
                        "expression": e,
                        "score": sc,
                        "error": None if sc is not None else "could not parse expression",
                    }
                    for e, sc in zip(exprs, scores)
                ]
            }
        if name == "get_feature_list":
            info = scorer.info()
            return {
                "features": scorer.feature_names,
                "operators": scorer.operators,
                "max_depth": info.max_depth,
                "model_type": info.model_type,
                "rules": [
                    "Every expression must reference at least one feature from `features`, spelled EXACTLY as listed (case-sensitive).",
                    "Numeric constants are NOT allowed — no literals of any kind.",
                    f"Keep tree depth <= {info.max_depth}.",
                    "Only `+ - * /` operators; no functions or comparisons.",
                ],
                "score_range": [0.5, 1.0],
                "higher_is_better": True,
            }
        return {"error": f"unknown tool: {name}"}
    except Exception as e:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Session log dataclass (persisted as JSONL per session)
# ---------------------------------------------------------------------------
@dataclass
class ToolCallLog:
    round_idx: int
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    latency_seconds: float


@dataclass
class SessionSummary:
    model: str
    rounds: int
    tool_calls: int
    unique_expressions_scored: int
    best_gat_score: Optional[float]
    best_expression: Optional[str]
    stopped_reason: str
    final_assistant_message: Optional[str]
    all_tool_calls: List[ToolCallLog] = field(default_factory=list)


def run_agentic_openai_loop(
    *,
    scorer: "GatScorerTool",
    model: str = "gpt-4o-mini",
    topic: str = _DEFAULT_TOPIC,
    max_rounds: int = 30,
    max_tool_calls: int = 150,
    temperature: float = 0.5,
    openai_client: Optional[Any] = None,
    output_dir: Optional[str] = None,
) -> SessionSummary:
    """Run an agentic OpenAI session where the GAT is exposed as a tool.

    Returns a :class:`SessionSummary` with the best expression found and the
    full tool-call log. If ``output_dir`` is provided, also writes:

      * ``<output_dir>/messages.jsonl`` — every message in the chat history.
      * ``<output_dir>/tool_calls.csv`` — flat table of tool calls + results.
      * ``<output_dir>/summary.json`` — the SessionSummary as JSON.
    """
    if openai_client is None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "The `openai` package is required for the OpenAI driver. "
                "Install with: pip install 'gat-agent-tool[openai]'"
            ) from e
        openai_client = OpenAI()

    tools = _tool_specs()

    system_prompt = (
        f"You are designing {topic}.\n"
        "A trained Graph Attention Network (GAT) is available to you as a tool "
        "that scores candidate expressions with a predicted AUC in [0.5, 1.0] "
        "(higher is better). Your goal is to find the highest-scoring expression.\n\n"
        "Strategy:\n"
        "  1. Call `get_feature_list` FIRST to learn the legal features and rules.\n"
        "  2. Propose a batch of candidates and call `score_expressions` on them (batch is much faster than single calls).\n"
        "  3. Read the scores; note which features and structures got high AUCs.\n"
        "  4. Propose a new, refined batch informed by the previous scores. Do not repeat expressions.\n"
        "  5. Continue until scores plateau or your budget runs out.\n\n"
        "When you are done, write a final message listing your top 5 expressions "
        "with their GAT scores, one per line, sorted by score descending."
    )
    user_prompt = (
        f"Please find a high-AUC expression. Your budget is at most "
        f"{max_tool_calls} tool calls and {max_rounds} turns. "
        "Start by calling get_feature_list."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    all_tool_calls: List[ToolCallLog] = []
    tool_call_count = 0
    unique_scored_exprs: Dict[str, float] = {}
    best_expr: Optional[str] = None
    best_score: Optional[float] = None
    final_text: Optional[str] = None
    stopped_reason = "reached max_rounds"

    for round_idx in range(int(max_rounds)):
        logger.info(
            "[round %d] calling %s (tool_calls_so_far=%d, messages=%d)",
            round_idx, model, tool_call_count, len(messages),
        )
        t0 = time.time()
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        logger.debug("[round %d] API latency: %.2fs", round_idx, time.time() - t0)

        msg = response.choices[0].message
        # Record the assistant turn verbatim so OpenAI can match tool_call_id on the next round.
        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            # Model produced a final message with no tool calls → we're done.
            final_text = msg.content
            stopped_reason = "model produced final answer"
            logger.info("[round %d] no tool calls → final answer received.", round_idx)
            break

        # Execute each tool call, append tool messages in order (OpenAI requires
        # every tool_call to be followed by a matching role='tool' message).
        any_budget_exceeded = False
        for tc in msg.tool_calls:
            if tool_call_count >= max_tool_calls:
                # Return a "budget exceeded" result rather than skipping — the
                # next iteration of the loop will notice and break out.
                result = {"error": f"tool call budget exceeded ({max_tool_calls})"}
                any_budget_exceeded = True
                latency = 0.0
            else:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    result = {"error": f"invalid JSON in tool arguments: {e}"}
                    latency = 0.0
                else:
                    t1 = time.time()
                    result = _run_tool(tc.function.name, args, scorer)
                    latency = time.time() - t1

                # Track the best score seen so far for the summary.
                if tc.function.name == "score_expression" and "score" in result and result["score"] is not None:
                    unique_scored_exprs[result["expression"]] = result["score"]
                    if best_score is None or result["score"] > best_score:
                        best_score, best_expr = result["score"], result["expression"]
                elif tc.function.name == "score_expressions":
                    for entry in result.get("results", []):
                        if entry.get("score") is not None:
                            unique_scored_exprs[entry["expression"]] = entry["score"]
                            if best_score is None or entry["score"] > best_score:
                                best_score, best_expr = entry["score"], entry["expression"]
                tool_call_count += 1

            # Log, append to message history.
            all_tool_calls.append(ToolCallLog(
                round_idx=round_idx,
                tool_name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}") if tc.function.arguments else {},
                result=result,
                latency_seconds=round(latency, 3),
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        if any_budget_exceeded:
            logger.warning(
                "[round %d] tool-call budget (%d) exhausted; requesting final summary.",
                round_idx, max_tool_calls,
            )
            messages.append({
                "role": "user",
                "content": (
                    "Your tool-call budget is exhausted. Produce your final message "
                    "now: list the top 5 expressions you have scored, sorted by GAT "
                    "score descending, one per line, with the score."
                ),
            })
            stopped_reason = "tool-call budget exhausted"
            # Do NOT break: the next round gives the model a chance to emit its
            # final summary message without calling any more tools.

    # One last request to get a clean final summary if we broke mid-conversation.
    if final_text is None and messages[-1]["role"] != "assistant":
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[],  # no tools — force a text answer
                temperature=temperature,
            )
            final_text = response.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            logger.warning("final-summary call failed: %s", e)

    summary = SessionSummary(
        model=model,
        rounds=round_idx + 1,
        tool_calls=tool_call_count,
        unique_expressions_scored=len(unique_scored_exprs),
        best_gat_score=best_score,
        best_expression=best_expr,
        stopped_reason=stopped_reason,
        final_assistant_message=final_text,
        all_tool_calls=all_tool_calls,
    )

    # ----- Optional persistence --------------------------------------
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "messages.jsonl").open("w") as f:
            for m in messages:
                f.write(json.dumps(m, default=str) + "\n")
        # Flat CSV of tool calls for quick inspection.
        try:
            import pandas as pd  # optional
            rows = [
                {
                    "round": tc.round_idx,
                    "tool": tc.tool_name,
                    "arguments_json": json.dumps(tc.arguments),
                    "result_json": json.dumps(tc.result),
                    "latency_seconds": tc.latency_seconds,
                }
                for tc in all_tool_calls
            ]
            pd.DataFrame(rows).to_csv(out / "tool_calls.csv", index=False)
        except ImportError:
            pass
        with (out / "summary.json").open("w") as f:
            json.dump(
                {
                    "model": summary.model,
                    "rounds": summary.rounds,
                    "tool_calls": summary.tool_calls,
                    "unique_expressions_scored": summary.unique_expressions_scored,
                    "best_gat_score": summary.best_gat_score,
                    "best_expression": summary.best_expression,
                    "stopped_reason": summary.stopped_reason,
                    "final_assistant_message": summary.final_assistant_message,
                },
                f, indent=2,
            )
        logger.info("wrote session artifacts under %s", out)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="gat-agent-tool loop-openai",
        description="Agentic iterative loop: OpenAI model uses the GAT as a function-calling tool.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to the GAT .pt checkpoint.")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--output_dir", default="gat_agent_runs/openai",
                   help="Directory for messages.jsonl, tool_calls.csv, summary.json.")
    p.add_argument("--topic", default=_DEFAULT_TOPIC)
    p.add_argument("--max_rounds", type=int, default=30)
    p.add_argument("--max_tool_calls", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var is required for loop-openai.")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Deferred import: torch_geometric is only needed once we actually run.
    from ..core import GatScorerTool
    scorer = GatScorerTool(args.checkpoint, device=args.device)

    summary = run_agentic_openai_loop(
        scorer=scorer,
        model=args.model,
        topic=args.topic,
        max_rounds=args.max_rounds,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )

    print(json.dumps({
        "model": summary.model,
        "rounds": summary.rounds,
        "tool_calls": summary.tool_calls,
        "unique_expressions_scored": summary.unique_expressions_scored,
        "best_gat_score": summary.best_gat_score,
        "best_expression": summary.best_expression,
        "stopped_reason": summary.stopped_reason,
        "output_dir": args.output_dir,
    }, indent=2))


if __name__ == "__main__":
    cli()
