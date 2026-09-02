from langgraph.graph import StateGraph, START, END
import psycopg_pool
from psycopg.rows import dict_row
from graphs.state import State
from graphs.nodes import entry, planner, researcher, coder, analyst, writer, reviewer, critic
from langgraph.checkpoint.postgres import PostgresSaver
from config.env import db_url

def planner_router(state: State) -> str:
    return str(state["next_agent"])

def reviewer_router(state: State) -> str:
    if state.get("final_output"):
        return "end"
    if state.get("status") == "accepted_at_limit":
        return "end"
    revision_count = state.get("revision_count") or 0
    if revision_count >= 2:
        return "end"
    return "writer"

graph = StateGraph(State)

graph.add_node("entry", entry)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("analyst", analyst)
graph.add_node("critic", critic)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_edge(START, "entry")
graph.add_edge("entry", "planner")

graph.add_conditional_edges(
    "planner",
    planner_router,
    {
        "researcher": "researcher",
        "coder": "coder",
        "analyst": "analyst",
        "critic": "critic",
        "end": "writer",
    }
)

graph.add_edge("researcher", "planner")
graph.add_edge("coder", "planner")
graph.add_edge("analyst", "planner")
graph.add_edge("critic", "planner")
graph.add_edge("writer", "reviewer")

graph.add_conditional_edges(
    "reviewer",
    reviewer_router,
    {
        "end": END,
        "writer": "writer",
    }
)

_pool = psycopg_pool.ConnectionPool(
    conninfo=db_url,
    min_size=1,
    max_size=5,
    kwargs={"autocommit": True, "row_factory": dict_row},
    open=True,
)

checkpointer = PostgresSaver(_pool)  # type: ignore
checkpointer.setup()
app = graph.compile(checkpointer=checkpointer)