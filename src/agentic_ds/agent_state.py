import json
from dataclasses import dataclass, field

@dataclass
class AgentState:
    user_request: str

    planned_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    remaining_tasks: list[str] = field(default_factory=list)

    completed_tools: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

    step: int = 0


def format_state(state: AgentState) -> str:
    return json.dumps(
        {
            "user_request": state.user_request,
            "planned_tasks": state.planned_tasks,
            "completed_tasks": state.completed_tasks,
            "remaining_tasks": state.remaining_tasks,
            "completed_tools": state.completed_tools,
            "tool_results": state.tool_results,
            "step": state.step,
        },
        indent=2,
    )
