import json
from ollama import chat

PLANNER_PROMPT = """
You are a planning component for a data science assistant.

Your job is to decompose the user's request into the minimum set of
information-gathering tasks required to answer it completely.

Do not execute tools.
Do not answer the user's question.
Do not invent information about the dataset.

Return ONLY valid JSON in this format:

{
  "tasks": [
    "<task 1>",
    "<task 2>"
  ]
}

Each task should describe an information need, not a specific tool call.

Keep the plan concise and avoid unnecessary tasks.
Each task must represent one distinct information need.
Do not combine multiple checks or analyses into a single task.
Tasks should be independently verifiable.
"""


def create_plan(user_request: str, model: str, temperature: float) -> tuple[list[str], str]:
    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT,
            },
            {
                "role": "user",
                "content": user_request,
            },
        ],
        format="json",
        options={"temperature": temperature},
    )

    planner_output = response.message.content

    try:
        plan = json.loads(planner_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Planner did not return valid JSON.\n"
            f"Raw output:\n{planner_output}"
        ) from exc

    if not isinstance(plan, dict):
        raise ValueError("Planner output must be a JSON object.")

    tasks = plan.get("tasks")

    if not isinstance(tasks, list) or not all(
        isinstance(task, str) for task in tasks
    ):
        raise ValueError(
            "Planner output must contain a 'tasks' list of strings."
        )

    return tasks, planner_output
