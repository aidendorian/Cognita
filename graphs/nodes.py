import json
from pathlib import Path
from rag.retriever import retrieve
from graphs.state import State
from app.llm import LLM, mask

llm = LLM()
VALID_AGENTS = {"researcher", "coder", "analyst", "writer", "reviewer", "end"}
MAX_REVISIONS = 3

def _append_message(state: State, role: str, content: str) -> list[dict]:
    messages = list(state.get("messages") or [])
    messages.append({"role": role, "content": content})
    return messages

def _scan_outputs(project_id: int) -> str:
    output_dir = Path(f"data/projects/{project_id}/outputs")
    if not output_dir.exists():
        return "None."
    files = [f.name for f in output_dir.iterdir() if f.is_file()]
    return ", ".join(files) if files else "None."

def entry(state: State) -> dict:
    messages = list(state.get("messages") or [])
    if not messages:
        messages.append({"role": "user", "content": state["task"]})
    return {
        "messages": messages,
        "revision_count": 0,
        "status": "running",
    }

def planner(state: State) -> dict:
    if state.get("final_output"):
        return {"next_agent": "end", "status": "complete"}

    prompt = f"""
    You are a research planning agent. Your job is to break down a task into clear steps
    and decide which specialist agent should act next based on what has already been done.

    TASK: {state["task"]}

    AVAILABLE AGENTS:
    - researcher : finds information, reads papers, does literature reviews, summarizes sources
    - coder      : writes and runs Python code, analyzes datasets, produces visualizations
    - analyst    : interprets findings, draws conclusions, identifies patterns and gaps
    - writer     : produces well-structured markdown reports and research documents
    - reviewer   : checks drafts for accuracy, missing citations, hallucinations, logical gaps
    - end        : use only when the task is fully complete and output is reviewed and ready

    CURRENT STATE:
    - Plan so far      : {state["plan"]}
    - Last step run    : {state["current_step"]}
    - Last status      : {state["status"]}
    - Research done    : {"yes" if state.get("research_summary") else "no"}
    - Code run         : {"yes" if state.get("code_result") else "no"}
    - Analysis done    : {"yes" if state.get("analysis_summary") else "no"}
    - Draft exists     : {"yes" if state.get("draft") else "no"}
    - Review notes     : {mask(state.get("review_notes"), limit=400) if state.get("review_notes") else "none"}
    - Output files     : {_scan_outputs(state["project_id"])}
    - Revision count   : {state.get("revision_count", 0)} of {MAX_REVISIONS} max

    INSTRUCTIONS:
    - Look at what has been completed before deciding the next step.
    - If status is "failed", decide whether to retry the last step or route to end.
    - If review notes exist and draft was rejected, route to writer (unless revision_count >= {MAX_REVISIONS}).
    - If revision_count >= {MAX_REVISIONS}, route to end — do not send back to writer again.
    - If draft exists and has not been reviewed, route to reviewer.
    - Only route to "end" when draft exists AND has been reviewed and approved.
    - Do not repeat a step that already produced output unless explicitly needed.

    Return ONLY a valid JSON object:
    {{
        "plan": ["step 1", "step 2", ...],
        "current_step": "the single next step to execute right now",
        "next_agent": "exactly one of: researcher, coder, analyst, writer, reviewer, end"
    }}
    """
    response = llm.generate(prompt)
    raw = str(response).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "plan": [],
            "current_step": "planning failed — could not parse LLM response",
            "next_agent": "end",
            "status": "failed",
            "messages": _append_message(state, "assistant", "[planner] failed to parse plan"),
        }

    if data.get("next_agent") not in VALID_AGENTS:
        data["next_agent"] = "researcher"

    return {
        "plan": data["plan"],
        "current_step": data["current_step"],
        "next_agent": data["next_agent"],
        "status": "running",
        "messages": _append_message(
            state, "assistant",
            f"[planner] next: {data['next_agent']} — {data['current_step']}"
        ),
    }

def researcher(state: State) -> dict:
    chunks = retrieve(str(state["current_step"]), project_id=state["project_id"], top_k=5)
    
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['chunk_text']}" for c in chunks) if chunks else "No relevant documents found in knowledge base."

    prompt = f"""
    You are a research specialist agent.

    OVERALL GOAL  : {state["task"]}
    CURRENT TASK  : {state["current_step"]}

    RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
    {context}

    PRIOR RESEARCH (build on this, do not repeat it):
    {mask(state.get("research_output"), limit=2000)}

    Using the retrieved context above as your primary source, produce a comprehensive 
    research report covering:
    1. Key concepts and definitions
    2. Current state of knowledge — important findings, methods, models
    3. Relevant papers, authors, or sources (cite from the retrieved context)
    4. Open questions and known gaps
    5. Technical details relevant to the overall goal

    Prioritize information from the retrieved context over general knowledge.
    Write clearly and thoroughly.
    """
    response = llm.generate(prompt)
    summary = llm.summarize(str(response))

    return {
        "research_output": response,
        "research_summary": summary,
        "status": "running",
        "messages": _append_message(state, "assistant", f"[researcher] {summary}"),
    }

def coder(state: State) -> dict:
    prompt = f"""
    You are an expert Python coding agent. You write clean, correct, well-commented Python code.

    OVERALL GOAL  : {state["task"]}
    CURRENT TASK  : {state["current_step"]}

    CONTEXT:
    - Research summary : {mask(state.get("research_summary"), limit=800)}
    - Analysis summary : {mask(state.get("analysis_summary"), limit=800)}
    - Previous code    : {mask(state.get("code_output"), limit=1000)}
    - Previous result  : {mask(state.get("code_result"), limit=600)}

    If previous code failed (check previous result for errors), fix the specific error.
    If no previous code exists, write fresh code for the current task.

    ENVIRONMENT:
    - Python 3.12
    - Available: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, pytest
    - Save any plots or output files to: outputs/
    - You cannot make network requests (no internet access in sandbox)

    Return ONLY raw Python code. No markdown fences, no explanation outside comments.
    """
    response = llm.generate(prompt)

    from tools.python_exec import PythonSandbox
    sandbox = PythonSandbox(project_id=state["project_id"])
    execution = sandbox.run(str(response))

    output_files = _scan_outputs(state["project_id"])

    if execution["success"]:
        result = (
            f"Code ran successfully.\n"
            f"Output:\n{execution['stdout']}\n"
            f"Files generated: {output_files}"
        )
    else:
        result = (
            f"Code failed (exit {execution['returncode']}).\n"
            f"Error:\n{execution['stderr']}"
        )

    return {
        "code_output": response,
        "code_result": result,
        "status": "running",
        "messages": _append_message(state, "assistant", f"[coder] {result[:200]}"),
    }

def analyst(state: State) -> dict:
    prompt = f"""
    You are an expert analyst agent. You interpret research findings, code results, and data
    to produce actionable insights and conclusions.

    OVERALL GOAL  : {state["task"]}
    CURRENT TASK  : {state["current_step"]}

    INPUTS TO ANALYZE:
    - Full research findings : {mask(state.get("research_output"), limit=2000)}
    - Code execution result  : {mask(state.get("code_result"), limit=800)}
    - Previous analysis      : {mask(state.get("analysis"), limit=800)}

    Produce a thorough analysis covering:
    1. Key patterns and insights from the research
    2. What the code output reveals (if available) — interpret numbers, charts, results
    3. Connections between findings — what do they mean together?
    4. Gaps, contradictions, or open questions
    5. Concrete recommendations for the writer — what should the report emphasize?

    Be specific. Avoid vague statements. This analysis directly shapes the final report.
    """
    response = llm.generate(prompt)
    summary = llm.summarize(str(response))

    return {
        "analysis": response,
        "analysis_summary": summary,
        "status": "running",
        "messages": _append_message(state, "assistant", f"[analyst] {summary}"),
    }

def writer(state: State) -> dict:
    revision_count = (state.get("revision_count") or 0) + 1

    prompt = f"""
    You are an expert research writing agent. You produce clear, well-structured,
    original research documents in markdown format.

    OVERALL GOAL  : {state["task"]}
    CURRENT TASK  : {state["current_step"]}
    REVISION      : {revision_count} of {MAX_REVISIONS} allowed

    CONTENT TO DRAW FROM:
    - Research summary   : {mask(state.get("research_summary"), limit=1000)}
    - Analysis summary   : {mask(state.get("analysis_summary"), limit=1000)}
    - Code result        : {mask(state.get("code_result"), limit=600)}
    - Output files       : {_scan_outputs(state["project_id"])}
    - Previous draft     : {mask(state.get("draft"), limit=1500)}
    - Reviewer feedback  : {state.get("review_notes") or "None yet — first draft."}

    INSTRUCTIONS:
    - If reviewer feedback exists, address EVERY point raised before anything else.
    - If no previous draft exists, write from scratch using the research and analysis.
    - Focus on originality, logical flow, and well-supported claims.
    - Every significant claim must be traceable to the research or analysis above.
    - Reference any output files by name where relevant (e.g. "see chart.png").
    - Use clear headings, subheadings, bullet points or tables where appropriate.

    Return ONLY valid markdown. No preamble, no commentary outside the document.
    """
    response = llm.generate(prompt)

    return {
        "draft": response,
        "revision_count": revision_count,
        "status": "running",
        "messages": _append_message(
            state, "assistant",
            f"[writer] draft revision {revision_count} complete"
        ),
    }

def reviewer(state: State) -> dict:
    revision_count = state.get("revision_count") or 0
    if revision_count >= MAX_REVISIONS:
        note = (
            f"AUTO-APPROVED after {MAX_REVISIONS} revision cycles. "
            "Draft accepted as-is to prevent infinite loop."
        )
        return {
            "review_notes": note,
            "final_output": state.get("draft"),
            "status": "complete",
            "messages": _append_message(state, "assistant", f"[reviewer] {note}"),
        }

    prompt = f"""
    You are an expert reviewer agent. You critically evaluate research drafts for quality,
    accuracy, and completeness before they are finalized.

    OVERALL GOAL  : {state["task"]}
    CURRENT TASK  : {state["current_step"]}
    REVISION      : {revision_count} of {MAX_REVISIONS} max (auto-approve at limit)

    MATERIALS TO REVIEW:
    - Draft            : {mask(state.get("draft"), limit=3000)}
    - Research summary : {mask(state.get("research_summary"), limit=600)}
    - Analysis summary : {mask(state.get("analysis_summary"), limit=600)}
    - Code result      : {mask(state.get("code_result"), limit=400)}
    - Output files     : {_scan_outputs(state["project_id"])}
    - Prior notes      : {mask(state.get("review_notes"), limit=400) or "None — first review."}

    CHECK FOR:
    1. Unsupported claims — facts not grounded in the research or analysis
    2. Missing citations — key findings that need sourcing
    3. Logical inconsistencies — conclusions that don't follow from evidence
    4. Contradictions with code results — claims that conflict with actual outputs
    5. Completeness — does the draft fully address the original task?
    6. Clarity — confusing sentences, undefined jargon

    FORMAT:
    - First line must be exactly "APPROVED" or "NEEDS REVISION"
    - If APPROVED: one paragraph explaining why it is ready
    - If NEEDS REVISION: numbered list of specific issues — each actionable,
    no vague feedback like "improve clarity"
    """
    response = llm.generate(prompt)
    approved = str(response).strip().upper().startswith("APPROVED")

    return {
        "review_notes": response,
        "final_output": state.get("draft") if approved else None,
        "status": "complete" if approved else "running",
        "messages": _append_message(
            state, "assistant",
            f"[reviewer] {'APPROVED' if approved else 'NEEDS REVISION'}"
        ),
    }