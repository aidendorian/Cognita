from config.llm import mask

def format_prior_memory(prior: dict, limit_per_item: int = 500) -> str:
    sections = []

    if prior.get("research_summaries"):
        items = [
            mask(s, limit=limit_per_item)
            for s in prior["research_summaries"]
        ]
        combined = "\n---\n".join(items)
        sections.append(f"PRIOR RESEARCH SESSIONS:\n{combined}")

    if prior.get("analysis_summaries"):
        items = [
            mask(s, limit=limit_per_item)
            for s in prior["analysis_summaries"]
        ]
        combined = "\n---\n".join(items)
        sections.append(f"PRIOR ANALYSIS SESSIONS:\n{combined}")
        
    if prior.get("critic_reports"):
        latest = mask(prior["critic_reports"][-1], limit=limit_per_item)
        sections.append(f"PRIOR NOVELTY ASSESSMENT:\n{latest}")

    if prior.get("prior_reports"):
        latest = mask(prior["prior_reports"][-1], limit=limit_per_item)
        sections.append(f"MOST RECENT REPORT:\n{latest}")

    if not sections:
        return ""

    return (
        "=" * 40 + "\n"
        "ACCUMULATED PROJECT MEMORY (from prior sessions)\n"
        "Use this to build on previous work, not repeat it.\n"
        + "=" * 40 + "\n\n"
        + "\n\n".join(sections)
        + "\n" + "=" * 40
    )

def has_prior_memory(prior: dict) -> bool:
    return any(
        prior.get(k)
        for k in ("research_summaries", "analysis_summaries", "prior_reports")
    )