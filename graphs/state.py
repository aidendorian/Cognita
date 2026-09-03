from typing_extensions import TypedDict
from typing import Optional
from typing import Annotated, Literal
from langgraph.graph.message import add_messages

class Evidence(TypedDict):
    id: str
    source: str
    url: Optional[str]
    excerpt: str
    status: Literal["unverified", "supported", "rejected"]
    reason: Optional[str]

class State(TypedDict):
    project_id: int
    run_id: Optional[str]
    task: str
    status: Optional[str]
    plan: Optional[list[str]]
    current_step: Optional[str]
    next_agent: Optional[str]

    research_output: Optional[str]
    research_summary: Optional[str]
    evidence: list[Evidence]

    code_output: Optional[str]
    code_result: Optional[str]
    analysis: Optional[str]
    analysis_summary: Optional[str]

    draft: Optional[str]
    review_notes: Optional[str]
    final_output: Optional[str]

    revision_count: Optional[int]
    messages: Annotated[list, add_messages]
    novelty_report: Optional[str]
    task_mode: Optional[str]
    output_files: Optional[list[str]]
    
def _merge_evidence(existing: list[Evidence], chunks: list[dict], web_results: list[dict]) -> list[Evidence]:
    evidence = list(existing)
    seen = {
        (e["source"], e.get("url"), e["excerpt"])
        for e in evidence
    }

    for chunk in chunks:
        item = {
            "source": chunk["source"],
            "url": chunk.get("url"),
            "excerpt": chunk["chunk_text"][:800],
        }

        key = (item["source"], item["url"], item["excerpt"])

        if key not in seen:
            data = {**item, "id": f"E{len(evidence)+1}", "status": "unverified", "reason": None}
            evidence.append(Evidence(**data))
            seen.add(key)

    for result in web_results:
        item = {
            "source": result.get("title") or result["url"],
            "url": result["url"],
            "excerpt": result.get("snippet", "")[:800],
        }

        key = (item["source"], item["url"], item["excerpt"])
        
        if key not in seen:
            data = {**item, "id": f"E{len(evidence)+1}", "status": "unverified", "reason": None}
            evidence.append(Evidence(**data))
            seen.add(key)

    return evidence

def _format_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(
        f"[{e['id']}] {e['source']}\n"
        f"{e['excerpt']}"
        for e in evidence
        if e["status"] != "rejected"
    )

def _render_references(evidence: list[Evidence]) -> str:
    usable = [
        e for e in evidence
        if e["status"] == "supported"
    ]

    if not usable:
        return ""

    lines = ["## References", ""]

    for e in usable:
        if e["url"]:
            lines.append(
                f'[{e["id"]}] {e["source"]} — {e["url"]}'
            )
        else:
            lines.append(
                f'[{e["id"]}] {e["source"]}'
            )

    return "\n".join(lines)