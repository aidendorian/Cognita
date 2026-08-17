from langgraph.graph import StateGraph, START, END
from graphs.state import State
from graphs.nodes import entry, planner, researcher, coder, analyst, writer, reviewer

def router(state: State) -> str:
    return str(state["next_agent"])


def reviewer_router(state: State) -> str:
    """
    Reviewer can either send the planner back for another loop,
    or end directly if it approved (final_output is set).
    This avoids one redundant planner call after approval.
    """
    if state.get("final_output"):
        return "end"
    return "planner"

graph = StateGraph(State)

graph.add_node("entry", entry)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_edge(START, "entry")
graph.add_edge("entry", "planner")

graph.add_conditional_edges(
    "planner",
    router,
    {
        "researcher": "researcher",
        "coder": "coder",
        "analyst": "analyst",
        "writer": "writer",
        "reviewer": "reviewer",
        "end": END,
    },
)

graph.add_edge("researcher", "planner")
graph.add_edge("coder", "planner")
graph.add_edge("analyst", "planner")
graph.add_edge("writer", "planner")

graph.add_conditional_edges(
    "reviewer",
    reviewer_router,
    {
        "planner": "planner",
        "end": END,
    },
)

app = graph.compile()