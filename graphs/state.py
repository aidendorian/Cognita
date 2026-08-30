from typing_extensions import TypedDict
from typing import Optional
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    project_id: int
    task: str
    status: Optional[str]

    plan: Optional[list[str]]
    current_step: Optional[str]
    next_agent: Optional[str]

    research_output: Optional[str]
    code_output: Optional[str]
    code_result: Optional[str]
    analysis: Optional[str]
    draft: Optional[str]
    review_notes: Optional[str]
    final_output: Optional[str]

    research_summary: Optional[str]
    analysis_summary: Optional[str]
    revision_count: Optional[int]
    messages: Annotated[list, add_messages]