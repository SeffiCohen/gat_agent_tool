"""Consumer-facing drivers that implement the propose-score-refine loop.

Two are provided out of the box:

  * :mod:`.local_hf` — loads a HuggingFace AutoModelForCausalLM locally and uses
    it as the proposer. No API key or network call needed.
  * :mod:`.openai_api` — uses the OpenAI API with native function-calling. The
    GAT is exposed to the model as a tool and the driver handles the tool-call
    round trips.

Both drivers share the same :class:`..core.GatScorerTool` for scoring and the
same :mod:`..validation` helpers for parsing proposals.
"""
