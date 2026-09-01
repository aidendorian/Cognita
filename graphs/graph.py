from langgraph.graph import StateGraph, START, END
import psycopg_pool
from psycopg.rows import dict_row
from graphs.state import State
from graphs.nodes import entry, planner, researcher, coder, analyst, writer, reviewer, critic
from langgraph.checkpoint.postgres import PostgresSaver
from config.env import db_url

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
graph.add_node("critic", critic)
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
        "critic": "critic",
        "end": END,
    },
)

graph.add_edge("researcher", "planner")
graph.add_edge("coder", "planner")
graph.add_edge("analyst", "planner")
graph.add_edge("writer", "planner")
graph.add_edge("critic", "planner")
graph.add_edge("writer", "reviewer")

graph.add_conditional_edges(
    "reviewer",
    reviewer_router,
    {
        "planner": "planner",
        "end": END,
    },
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