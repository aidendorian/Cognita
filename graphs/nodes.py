import json
from pathlib import Path
from rag.retriever import retrieve
from graphs.state import State
from app.llm import LLM, mask
from tools.filesystem import FileSystem
from memory.project import load_project_memory
from memory.summaries import format_prior_memory
from rag.ingest import ingest
from tools.python_exec import PythonSandbox
from tools.search import search as web_search
from pathlib import Path
import re

llm = LLM()
VALID_AGENTS = {"researcher", "coder", "analyst", "writer", "reviewer", "end"}
MAX_REVISIONS = 2

def _multi_retrieve(queries: list[str], project_id: int, top_k: int = 5) -> list[dict]:
    seen = set()
    results = []
    for query in queries:
        if not query:
            continue
        for chunk in retrieve(query, project_id=project_id, top_k=top_k):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                results.append(chunk)
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k * 2]

def _extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found", text, 0)

def _scan_outputs(project_id: int) -> str:
    output_dir = Path(f"data/projects/{project_id}/outputs")
    if not output_dir.exists():
        return "None."
    files = [f.name for f in output_dir.iterdir() if f.is_file()]
    return ", ".join(files) if files else "None."

def entry(state: State) -> dict:
    return {
        "messages": [{"role": "user", "content": state["task"]}],
        "revision_count": 0,
        "status": "running",
    }

def planner(state: State) -> dict:
    if state.get("final_output"):
        return {"next_agent": "end", "status": "complete"}

    prompt = f"""
    CRITICAL: Respond with ONLY a JSON object. No explanation, no markdown, no text before or after. Start your response with {{ and end with }}.
    You are a research planning agent. Your job is to break down a task into clear steps
    and decide which specialist agent should act next based on what has already been done.

    TASK: {state["task"]}

    AVAILABLE AGENTS:
    - researcher : finds information, reads papers, does literature reviews, summarizes sources
    - coder : writes and runs Python code, analyzes datasets, produces visualizations
    - analyst : interprets findings, draws conclusions, identifies patterns and gaps
    - writer : produces well-structured markdown reports and research documents
    - reviewer : checks drafts for accuracy, missing citations, hallucinations, logical gaps
    - end : use only when the task is fully complete and output is reviewed and ready

    CURRENT STATE:
    - Plan so far : {state["plan"]}
    - Last step run : {state["current_step"]}
    - Last status : {state["status"]}
    - Research done : {"yes" if state.get("research_summary") else "no"}
    - Code run : {"yes" if state.get("code_result") else "no"}
    - Analysis done : {"yes" if state.get("analysis_summary") else "no"}
    - Draft exists : {"yes" if state.get("draft") else "no"}
    - Review notes : {mask(state.get("review_notes"), limit=400) if state.get("review_notes") else "none"}
    - Output files : {_scan_outputs(state["project_id"])}
    - Revision count : {state.get("revision_count", 0)} of {MAX_REVISIONS} max

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
    response = llm.generate(prompt, max_output_tokens=8196)
    try:
        data = _extract_json(str(response))
    except json.JSONDecodeError:
        return {
            "plan": [],
            "current_step": "planning failed — could not parse LLM response",
            "next_agent": "end",
            "status": "failed",
            "messages": [{"role": "assistant", "content": "[planner] failed to parse plan"}],
        }

    if data.get("next_agent") not in VALID_AGENTS:
        data["next_agent"] = "researcher"

    return {
        "plan": data["plan"],
        "current_step": data["current_step"],
        "next_agent": data["next_agent"],
        "status": "running",
        "messages": [{"role": "assistant", "content": f"[planner] next: {data['next_agent']} — {data['current_step']}"}],
    }

def researcher(state: State) -> dict:
    
    prior = load_project_memory(state["project_id"])    
    prior_context = format_prior_memory(prior, limit_per_item=300)
    
    chunks = _multi_retrieve(
        queries=[str(state["current_step"]),
                str(state["task"]),
                mask(state.get("research_summary", ""), limit=200),],project_id=state["project_id"]
    )
    
    search_results = []
    try:
        search_results = web_search(
            query=str(state["current_step"]),
            project_id=state["project_id"],
            max_results=3
        )
    except Exception as e:
        print(f"[researcher] web search failed: {e}")
    web_context = "\n\n".join(f"[Web: {r.get('url', '')}]\n{r.get('content', '')[:300]}" for r in search_results) if search_results else "No web results."
    
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['chunk_text'][:300]}" for c in chunks) if chunks else "No relevant documents found in knowledge base."

    prompt = f"""
    You are a research specialist agent.

    OVERALL GOAL : {state["task"]}
    CURRENT TASK : {state["current_step"]}

    RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
    {context}

    FRESH WEB SEARCH RESULTS:
    {web_context}

    PRIOR RESEARCH (build on this, do not repeat it):
    {mask(state.get("research_output"), limit=1000)}
    
    ACCUMULATED KNOWLEDGE FROM PRIOR SESSIONS:
    {prior_context if prior_context else "No prior sessions yet — this is the first run."}
    
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
    response = llm.generate(prompt, max_output_tokens=8196)
    summary = llm.summarize(str(response), max_words=8196) #or mask(str(response), limit=1500)

    return {
        "research_output": response,
        "research_summary": summary,
        "status": "running",
        "messages": [{"role": "assistant", "content": f"[researcher] {summary}"}],
    }

def coder(state: State) -> dict:
    prompt = f"""
    You are an expert Python coding agent. You write clean, correct, well-commented Python code.

    OVERALL GOAL : {state["task"]}
    CURRENT TASK : {state["current_step"]}

    CONTEXT:
    - Research summary : {mask(state.get("research_summary"), limit=800)}
    - Analysis summary : {mask(state.get("analysis_summary"), limit=800)}
    - Previous code : {mask(state.get("code_output"), limit=1000)}
    - Previous result : {mask(state.get("code_result"), limit=600)}

    If previous code failed (check previous result for errors), fix the specific error.
    If no previous code exists, write fresh code for the current task.

    ENVIRONMENT:
    - Python 3.12
    - Available: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, pytest
    - Save any plots or output files to: outputs/
    - You cannot make network requests (no internet access in sandbox)

    Return ONLY raw Python code. No markdown fences, no explanation outside comments.
    """
    
    MAX_CODE_RETRIES = 3

    code = str(llm.generate(prompt, max_output_tokens=8192))
    sandbox = PythonSandbox(project_id=state["project_id"])
    execution = sandbox.run(code)

    attempt = 0
    while not execution["success"] and attempt < MAX_CODE_RETRIES:
        attempt += 1
        fix_prompt = f"""
        This Python code failed. Fix the specific error and return ONLY corrected Python code.

        ATTEMPT: {attempt} of {MAX_CODE_RETRIES}
        ERROR:
        {execution['stderr'][:800]}

        FAILING CODE:
        {code}

        Return ONLY raw Python code. No markdown, no explanation.
        """
        
        code = str(llm.generate(fix_prompt, max_output_tokens=8192))
        execution = sandbox.run(code)
    
    output_files = _scan_outputs(state["project_id"])
    outputs_path = Path(f"data/projects/{state['project_id']}/outputs")
    if outputs_path.exists():
        for file in outputs_path.iterdir():
            if file.suffix in (".csv", ".txt", ".json", ".md"):
                try:
                    ingest(str(file), project_id=state["project_id"], source_type="text")
                except Exception as e:
                    print(f"[coder] failed to ingest {file.name}: {e}")
                    
    if execution["success"]:
        result = (
            f"Code ran successfully after {attempt} fix(es).\n"
            f"Output:\n{execution['stdout']}\n"
            f"Files: {output_files}"
        )
    else:
        result = (
            f"Code failed after {MAX_CODE_RETRIES} attempts.\n"
            f"Final error:\n{execution['stderr'][:500]}\n"
            f"The planner will decide whether to retry or continue."
        )

    return {
        "code_output": code,
        "code_result": result,
        "status": "running",
        "messages": [{"role": "assistant", "content": f"[coder] {result[:200]}"}],
    }

def analyst(state: State) -> dict:
    prior = load_project_memory(state["project_id"])
    prior_context = format_prior_memory(prior)
    fs = FileSystem(project_id=state["project_id"])
    csv_content = ""
    for f in fs.list_files("outputs"):
        if f.endswith(".csv"):
            try:
                content = fs.read(f)
                csv_content += f"\n[{f}]\n{content[:1500]}"
            except Exception:
                pass
    
    prompt = f"""
    You are an expert analyst agent. You interpret research findings, code results, and data
    to produce actionable insights and conclusions.

    OVERALL GOAL : {state["task"]}
    CURRENT TASK : {state["current_step"]}

    INPUTS TO ANALYZE:
    - Full research findings : {mask(state.get("research_output"), limit=2000)}
    - Code execution result : {mask(state.get("code_result"), limit=800)}
    - Previous analysis : {mask(state.get("analysis"), limit=800)}
    - CSV data files: {csv_content or "None."}

    ACCUMULATED KNOWLEDGE FROM PRIOR SESSIONS:
    {prior_context if prior_context else "No prior sessions yet — this is the first run."}
    
    Produce a thorough analysis covering:
    1. Key patterns and insights from the research
    2. What the code output reveals (if available) — interpret numbers, charts, results
    3. Connections between findings — what do they mean together?
    4. Gaps, contradictions, or open questions
    5. Concrete recommendations for the writer — what should the report emphasize?

    Be specific. Avoid vague statements. This analysis directly shapes the final report.
    """
    response = llm.generate(prompt, max_output_tokens=8196)
    summary = llm.summarize(str(response)) or mask(str(response), limit=500)

    return {
        "analysis": response,
        "analysis_summary": summary,
        "status": "running",
        "messages": [{"role": "assistant", "content": f"[analyst] {summary}"}],
    }

def writer(state: State) -> dict:
    revision_count = (state.get("revision_count") or 0) + 1

    prompt = f"""
    You are an expert research writing agent. You produce clear, well-structured,
    original research documents in markdown format.

    OVERALL GOAL : {state["task"]}
    CURRENT TASK : {state["current_step"]}
    REVISION : {revision_count} of {MAX_REVISIONS} allowed

    CONTENT TO DRAW FROM:
    - Research summary : {mask(state.get("research_summary"), limit=1000)}
    - Analysis summary : {mask(state.get("analysis_summary"), limit=1000)}
    - Code result : {mask(state.get("code_result"), limit=600)}
    - Output files : {_scan_outputs(state["project_id"])}
    - Previous draft : {state.get("draft") or "None yet."}
    - Reviewer feedback : {state.get("review_notes") or "None yet — first draft."}

    INSTRUCTIONS:
    - If reviewer feedback exists, this is a REVISION — read every numbered point carefully
    - Fix ONLY the exact sections mentioned — reproduce everything else unchanged
    - For mathematical notation: ensure subscripts and operations are consistent throughout
    - After fixes, add at the very top before the title:
        Revision {revision_count}: Fixed [list exactly what you fixed]
    - Then reproduce the COMPLETE document with fixes applied
    - If no reviewer feedback, write from scratch using research and analysis
    - Every significant claim must be traceable to the research or analysis above
    - Reference output files by name where relevant
    - Use clear headings, subheadings, bullet points or tables
    Return ONLY valid markdown. No preamble, no commentary outside the document.
    """
    response = llm.generate(prompt)
    fs = FileSystem(project_id=state["project_id"])
    filename = f"outputs/draft_v{revision_count}.md"
    fs.write(filename, str(response))
    
    return {
        "draft": response,
        "revision_count": revision_count,
        "status": "running",
        "messages": [{"role": "assistant", "content": f"[writer] draft revision {revision_count} complete — saved to {filename}"}],
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
            "messages": [{"role": "assistant", "content": f"[reviewer] {note}"}],
        }

    prompt = f"""
    You are an expert reviewer agent. You critically evaluate research drafts for quality,
    accuracy, and completeness before they are finalized.

    OVERALL GOAL : {state["task"]}
    CURRENT TASK : {state["current_step"]}
    REVISION : {revision_count} of {MAX_REVISIONS} max (auto-approve at limit)

    MATERIALS TO REVIEW:
    - Draft : {state.get("draft") or "No draft yet."}
    - Research summary : {mask(state.get("research_summary"), limit=600)}
    - Analysis summary : {mask(state.get("analysis_summary"), limit=600)}
    - Code result : {mask(state.get("code_result"), limit=400)}
    - Output files : {_scan_outputs(state["project_id"])}
    - Prior notes : {mask(state.get("review_notes"), limit=400) or "None — first review."}
    
    CONTEXT:
    - This is a research summary report, not a peer-reviewed journal submission
    - Approve if the document is substantially correct, complete, and addresses the task
    - Only flag genuinely wrong facts or broken logic — not stylistic preferences
    - Mathematical notation only needs to be internally consistent, not publication-perfect
    - If the writer addressed the previous revision notes, do not re-flag the same issues
    - When in doubt, APPROVE — a good summary is better than an infinite revision loop

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
    response = llm.generate(prompt, max_output_tokens=8196)
    approved = str(response).strip().upper().startswith("APPROVED")
    if approved:
        fs = FileSystem(project_id=state["project_id"])
        fs.write("outputs/final_report.md", state.get("draft", "")) #type: ignore
    return {
        "review_notes": response,
        "final_output": state.get("draft") if approved else None,
        "status": "complete" if approved else "running",
        "messages": [{"role": "assistant", "content": f"[reviewer] {'APPROVED' if approved else 'NEEDS REVISION'}"}],
    }