# Lab Report — Day 08 LangGraph Agentic Orchestration

## 1. Metrics Summary

| Metric | Value |
|--------|-------|
| Total Scenarios | 7 |
| Success Rate | 100.0% |
| Avg Nodes Visited | 6.4 |
| Total Retries | 3 |
| Total Interrupts (HITL) | 2 |
| Resume Success | False |

## 2. Per-Scenario Results

| Scenario ID | Expected Route | Actual Route | Success | Nodes | Retries | Approval |
|-------------|---------------|--------------|---------|-------|---------|----------|
| S01_simple | simple | simple | ✅ | 4 | 0 | — |
| S02_tool | tool | tool | ✅ | 6 | 0 | — |
| S03_missing | missing_info | missing_info | ✅ | 4 | 0 | — |
| S04_risky | risky | risky | ✅ | 8 | 0 | ✅ |
| S05_error | error | error | ✅ | 10 | 2 | — |
| S06_delete | risky | risky | ✅ | 8 | 0 | ✅ |
| S07_dead_letter | error | error | ✅ | 5 | 1 | — |

## 3. Architecture

### Graph Design
The workflow is a `StateGraph` with 11 nodes wired in a directed graph with conditional routing:

```
START → intake → classify → [route_after_classify]
  simple       → answer → finalize → END
  tool         → tool → evaluate → [route_after_evaluate]
                   success  → answer → finalize → END
                   retry    → retry  → [route_after_retry]
                                tool (loop back) or dead_letter → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → [route_after_approval]
                                  approved → tool → evaluate → ...
                                  rejected → clarify → finalize → END
  error        → retry → [route_after_retry] → tool or dead_letter
```

### State Schema
`AgentState` uses `TypedDict` with:
- **Overwrite fields**: `route`, `risk_level`, `attempt`, `final_answer`, `evaluation_result`, `pending_question`, `proposed_action`, `approval`
- **Append-only reducers** (`Annotated[list, add]`): `messages`, `tool_results`, `errors`, `events`

Additional fields added beyond the starter: `evaluation_result`, `pending_question`, `proposed_action`, `approval`.

### LLM Integration
- `classify_node`: Uses `llm.with_structured_output(ClassificationResult)` for reliable intent classification via Pydantic model.
- `answer_node`: Uses LLM with grounded context (query + tool_results + approval) to generate responses.

## 4. Failure Analysis

### Failure Mode 1: LLM Classification Errors
The `classify_node` relies on an LLM which may occasionally misclassify borderline queries (e.g., treating a vague refund request as `missing_info` instead of `risky`). This is mitigated by the priority prompt (risky > tool > missing_info > error > simple) and structured output via Pydantic.

### Failure Mode 2: Unbounded Retry Loops
Without the `attempt < max_attempts` bound in `route_after_retry`, error scenarios would loop forever. The `max_attempts` field (default 3, overridable per scenario) gates the loop and routes to `dead_letter` on exhaustion.

### Failure Mode 3: Missing HITL on Approval Path
If `approval_node` fails or approval dict is malformed, `route_after_approval` defaults to `clarify` rather than crashing. The mock auto-approval ensures CI passes while real HITL is opt-in via `LANGGRAPH_INTERRUPT=true`.

## 5. Improvement Plan

1. **LLM-as-judge in evaluate_node**: Replace the `ERROR` substring heuristic with an LLM call to judge tool result quality.
2. **Parallel fan-out**: Use `Send()` to call multiple tools concurrently for complex queries.
3. **Real HITL UI**: Build a Streamlit dashboard that receives `interrupt()` payloads and lets a human approve/reject.
4. **Time travel**: Use `get_state_history()` to replay scenarios for debugging.
5. **Crash recovery**: Swap `MemorySaver` for `SqliteSaver` so checkpoints survive process restarts.
