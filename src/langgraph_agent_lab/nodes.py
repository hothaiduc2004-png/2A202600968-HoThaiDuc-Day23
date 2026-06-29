"""Node functions for the LangGraph workflow."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from .state import AgentState, Route, make_event


# ─── Helper: load env ─────────────────────────────────────────────────
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


# ─── Structured output model for classification ───────────────────────
class ClassificationResult(BaseModel):
    route: str
    risk_level: str
    reasoning: str


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    _load_env()
    from .llm import get_llm

    query = state.get("query", "")

    llm = get_llm(temperature=0.0)

    system_prompt = """You are a support ticket classifier. Classify the user query into exactly one route.

Routes and priority (highest first):
1. risky   - Actions with side effects: refunds, deletions, sending emails, cancellations, account modifications, any destructive or irreversible action
2. tool    - Information lookups: order status, tracking, search queries, data retrieval
3. missing_info - Vague or incomplete queries lacking actionable context (e.g. "fix it", "help me", "it's broken")
4. error   - System failures: timeouts, crashes, service unavailable, technical errors
5. simple  - General questions answerable without tools (e.g. how-to, FAQ, password reset instructions)

Return a JSON with:
- route: one of "simple", "tool", "missing_info", "risky", "error"
- risk_level: "high" if route is "risky", otherwise "low"
- reasoning: brief explanation"""

    structured_llm = llm.with_structured_output(ClassificationResult)
    
    result: ClassificationResult = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify this support ticket: {query}"},
    ])

    route = result.route if result.route in [r.value for r in Route] else Route.SIMPLE.value
    risk_level = result.risk_level if result.risk_level in ("high", "low") else (
        "high" if route == Route.RISKY.value else "low"
    )

    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:route={route}"],
        "events": [make_event("classify", "completed", f"classified as {route}", reasoning=result.reasoning)],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with simulated transient failures for error routes."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)

    # Simulate transient failure for error-route scenarios on first 2 attempts
    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: Tool call failed on attempt {attempt + 1} — simulated transient timeout"
        return {
            "tool_results": [result],
            "events": [make_event("tool", "error", f"tool failed attempt {attempt + 1}", attempt=attempt)],
        }

    # Success case
    query = state.get("query", "")
    result = f"TOOL_SUCCESS: Retrieved data for query '{query[:60]}' — mock result: {{status: 'ok', data: 'order_12345_shipped'}}"
    return {
        "tool_results": [result],
        "messages": [f"tool:{result[:60]}"],
        "events": [make_event("tool", "completed", "tool call succeeded", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — retry-loop gate."""
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""

    # Heuristic: check for ERROR substring
    if "ERROR" in latest.upper():
        evaluation_result = "needs_retry"
        message = "tool result indicates failure, retrying"
    else:
        evaluation_result = "success"
        message = "tool result is satisfactory"

    return {
        "evaluation_result": evaluation_result,
        "events": [make_event("evaluate", "completed", message, evaluation=evaluation_result)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in available context."""
    _load_env()
    from .llm import get_llm

    query = state.get("query", "")
    tool_results = state.get("tool_results") or []
    approval = state.get("approval")
    route = state.get("route", "")

    context_parts = [f"User query: {query}"]
    if tool_results:
        context_parts.append(f"Tool results: {'; '.join(tool_results[-3:])}")
    if approval:
        approved = approval.get("approved", False)
        comment = approval.get("comment", "")
        context_parts.append(f"Approval decision: {'Approved' if approved else 'Rejected'}. {comment}")
    context_parts.append(f"Route: {route}")

    context = "\n".join(context_parts)

    llm = get_llm(temperature=0.2)
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful customer support agent. Provide a clear, concise, and helpful response to the user based on the available context."},
        {"role": "user", "content": context},
    ])

    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    
    question = (
        f"I'd be happy to help, but I need more details. "
        f"Your request '{query[:80]}' is too vague. "
        f"Could you please provide: (1) What specific issue are you experiencing? "
        f"(2) Which product or service is affected? "
        f"(3) What have you already tried?"
    )
    final_answer = question

    return {
        "pending_question": question,
        "final_answer": final_answer,
        "events": [make_event("clarify", "completed", "clarification question generated")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    route = state.get("route", "")
    
    proposed_action = (
        f"PROPOSED ACTION: '{query}'\n"
        f"Route: {route} | Risk level: HIGH\n"
        f"This action has irreversible side effects and requires explicit human approval "
        f"before execution. Please review and approve or reject."
    )

    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "pending_approval", "risky action prepared for approval", query=query)],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step. Default: mock approval."""
    use_interrupt = os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true"
    
    if use_interrupt:
        from langgraph.types import interrupt
        proposed = state.get("proposed_action", "")
        decision = interrupt({"proposed_action": proposed, "message": "Approve this action? (yes/no)"})
        approved = str(decision).lower() in ("yes", "true", "approve", "1")
        comment = f"Human decision: {decision}"
    else:
        # Mock: auto-approve for CI/testing
        approved = True
        comment = "Auto-approved by mock reviewer"

    approval: dict[str, Any] = {
        "approved": approved,
        "reviewer": "mock-reviewer" if not use_interrupt else "human-reviewer",
        "comment": comment,
    }

    return {
        "approval": approval,
        "events": [make_event("approval", "completed", f"approval={'granted' if approved else 'rejected'}", approved=approved)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt — increment counter and log failure."""
    attempt = state.get("attempt", 0)
    new_attempt = attempt + 1
    error_msg = f"Attempt {new_attempt} failed — will retry or escalate to dead letter"

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "retry_attempt", error_msg, attempt=new_attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    query = state.get("query", "")
    errors = state.get("errors") or []

    final_answer = (
        f"We were unable to process your request after {attempt} attempt(s). "
        f"Your ticket has been escalated to our engineering team for manual review. "
        f"Error summary: {'; '.join(errors[-3:]) if errors else 'Max retries exceeded'}."
    )

    return {
        "route": Route.ERROR.value,  # keep original route for metric matching; dead_letter is recorded via events
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter", "escalated",
                f"dead letter after {attempt}/{max_attempts} attempts",
                query=query[:80],
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    route = state.get("route", "unknown")
    scenario_id = state.get("scenario_id", "unknown")
    has_answer = bool(state.get("final_answer") or state.get("pending_question"))

    return {
        "events": [
            make_event(
                "finalize", "completed", "workflow finished",
                route=route,
                scenario_id=scenario_id,
                has_answer=has_answer,
            )
        ],
    }
