import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from data_tools import (describe_numeric_columns, inspect_dataframe,
                        inspect_missing_values, load_dataset)
from ollama import chat

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"agent_run_{RUN_ID}.log"


def log_event(title: str, content: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n--- {title} ---\n")
        f.write(content)
        f.write("\n")


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

def create_plan(user_request: str) -> list[str]:
    response = chat(
        model=MODEL,
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
    )

    planner_output = response.message.content

    log_event("PLANNER RAW OUTPUT", planner_output)

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

    return tasks


"""
Agentic Data Science Assistant
Concepts implemented:
- tool calls
- looping through multiple steps
- maintaining agent state across steps
- using agent state to determine next actions
- planning component to decompose user requests into tasks
This is a more advanced version of the data tool loop.
"""

# MODEL = "qwen3:4b-q4_K_M" # 3.3 GB    27%/73% CPU/GPU
# MODEL = "llama3.2:3b-instruct-q5_K_M" # 3.2 GB    24%/76% CPU/GPU
# MODEL = "phi4-mini:3.8b-q4_K_M" # 3.1 GB    24%/76% CPU/GPU

MODEL = "qwen2.5:3b-instruct-q4_K_M" # 2.2 GB    100% GPU
# MODEL = "qwen2.5:1.5b" # 1.2 GB    100% GPU


TOOLS = {
    "inspect_dataframe": inspect_dataframe,
    "inspect_missing_values": inspect_missing_values,
    "describe_numeric_columns": describe_numeric_columns,
}


SYSTEM_PROMPT = """
You are a data science assistant with access to tools.

Available tools:

1. inspect_dataframe
   Description:
   Inspect the structure of the currently loaded pandas DataFrame.

   Arguments:
    - None

   Returns:
   - number of rows
   - number of columns
   - column names
   - counts of pandas data types
   - approximate memory usage

2. inspect_missing_values
   Description:
   Inspect missing values in the currently loaded pandas DataFrame.

   Arguments:
    - None

   Returns:
   - columns containing missing values
   - missing count for each affected column
   - total number of missing values

3. describe_numeric_columns
   Description:
   Compute descriptive statistics for numerical columns.

   Arguments:
   - columns: optional list[str]
     Names of the numerical columns to describe.
     If omitted, all numerical columns are described.

   Returns:
   - total number of numerical columns
   - columns that were described
   - count, mean, standard deviation, minimum,
     quartiles, median, and maximum

You may use more than one tool if necessary.

If you need to use a tool, respond ONLY with valid JSON:

{
  "type": "tool_call",
  "tool": "<tool_name>",
  "arguments": {}
}

For a tool with arguments:

{
  "type": "tool_call",
  "tool": "describe_numeric_columns",
  "arguments": {
    "columns": ["population", "medIncome"]
  }
}

If you already have enough information to answer ALL parts of the
user's request, respond ONLY with valid JSON:

{
  "type": "final_answer",
  "answer": "<your answer>"
}

When calling a tool, identify which planned task or tasks the call
is intended to address.

The "tasks" field MUST be a JSON list.

Every item in "tasks" MUST exactly match one of the strings currently
listed in remaining_tasks.

Use this format:

{
  "type": "tool_call",
  "tool": "<tool_name>",
  "arguments": {},
  "tasks": [
    "<exact planned task being addressed>"
  ]
}

Important rules:

- Base conclusions only on information explicitly available from tool results.
- Do not assume facts that have not been observed.
- Before giving a final answer, check whether every part of the user's request has been answered.
- If relevant information is still missing, use another appropriate tool.
- Do not invent information about the dataset.

The current agent state will be provided after each tool execution.

Use the state to determine:
- what information has already been obtained;
- which tools have already been executed;
- whether additional information is needed.

Do not repeat a tool unless there is a clear reason to do so.
"""


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


def main() -> None:
    df = load_dataset()

    print("\nExamples of questions you can ask:")
    print("- How many rows and columns are in the dataset?")
    print("- Which columns contain missing values?")
    print(
        "- Tell me the dataset dimensions and which columns "
        "contain missing values."
    )
    print("- Give me descriptive statistics for population and medIncome.")
    print("- What are the typical values and ranges of population and householdsize?")
    print(
        "- Tell me the dataset dimensions, identify missing data, "
        "and summarize population and medIncome."
    )
    print("- Compare the descriptive statistics of population and medIncome.")
    print("- Give me a concise assessment of the dataset structure, data completeness, and the typical values of population and medIncome.")
    print()

    user_message = input("User: ")
    log_event("USER REQUEST", user_message)
    planned_tasks = create_plan(user_message)

    state = AgentState(
        user_request=user_message,
        planned_tasks=planned_tasks,
        remaining_tasks=planned_tasks.copy(),
    )

    print("\n--- PLAN ---\n")

    for i, task in enumerate(state.planned_tasks, start=1):
        print(f"{i}. {task}")


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{user_message}\n\n"
                "Current agent state:\n"
                + format_state(state)
                + "\n\nDecide the next action."
            ),
        },
    ]

    # Loop through multiple steps, allowing the agent to call tools and update its state
    max_steps = 5

    for step in range(1, max_steps + 1):
        response = chat(
            model=MODEL,
            messages=messages,
            format="json",
        )

        llm_output = response.message.content

        log_event(
            f"LLM DECISION {step}",
            llm_output,
        )

        try:
            decision = json.loads(llm_output)
            print(f"\n--- LLM DECISION {step} ---\n")
            print(llm_output)
        except json.JSONDecodeError:
            if state.remaining_tasks:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response did not follow the required JSON protocol, "
                            "and planned tasks are still incomplete:\n"
                            + json.dumps(state.remaining_tasks, indent=2)
                            + "\nReturn the next action using the required JSON format."
                        ),
                    }
                )
                continue

            print("\n--- FINAL ANSWER ---\n")
            print(llm_output)
            return


        if decision["type"] == "final_answer":
            if state.remaining_tasks:
                messages.append(
                    {
                        "role": "assistant",
                        "content": llm_output,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You attempted to give a final answer, "
                            "but the following planned tasks are still incomplete:\n"
                            + json.dumps(state.remaining_tasks, indent=2)
                            + "\nContinue working on the remaining tasks."
                        ),
                    }
                )
                continue

            print("\n--- FINAL ANSWER ---\n")
            print(decision["answer"])
            return

        if decision["type"] != "tool_call":
            raise ValueError(
                f"Unknown decision type: {decision['type']}"
            )

        tool_name = decision["tool"]

        if tool_name not in TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")

        arguments = decision.get("arguments", {})

        if not isinstance(arguments, dict):
            raise ValueError(
                f"'arguments' must be an object, got {type(arguments).__name__}"
            )

        # Validate tool arguments before execution
        tool_function = TOOLS[tool_name]
        signature = inspect.signature(tool_function)

        valid_arguments = {
            name
            for name in signature.parameters
            if name != "df"
        }

        invalid_arguments = [
            name
            for name in arguments
            if name not in valid_arguments
        ]

        if invalid_arguments:
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_output,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The tool '{tool_name}' does not accept these arguments: "
                        f"{invalid_arguments}.\n"
                        f"Valid arguments are: {sorted(valid_arguments)}.\n"
                        "Return a corrected tool call using the required JSON format."
                    ),
                }
            )
            continue


        # Validation. We do not trust the LLM to always follow the schema.
        tasks = decision.get("tasks", [])

        if not isinstance(tasks, list):
            raise ValueError(
                f"'tasks' must be a list, got {type(tasks).__name__}"
            )

        if not tasks:
            raise ValueError(
                "'tasks' must not be empty for a tool call."
            )

    
        # Also validate that he didn't invent tasks:
        invalid_tasks = [
            task
            for task in tasks
            if task not in state.remaining_tasks
        ]

        if invalid_tasks:
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_output,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The following tasks are not valid remaining tasks:\n"
                        + json.dumps(invalid_tasks, indent=2)
                        + "\n\nCurrent remaining tasks:\n"
                        + json.dumps(state.remaining_tasks, indent=2)
                        + "\nChoose an action only for a remaining task."
                    ),
                }
            )
            continue

        tool_result = tool_function(
            df,
            **arguments,
        )


        log_event(
            f"TOOL RESULT {step}: {tool_name}",
            json.dumps(tool_result, indent=2),
        )


        state.step = step
        state.completed_tools.append(tool_name)

        state.tool_results.append(
            {
        "step": step,
        "tool": tool_name,
        "arguments": arguments,
        "result": tool_result,
            }
        )


        for task in tasks:
            if task in state.remaining_tasks:
                state.remaining_tasks.remove(task)
                state.completed_tasks.append(task)

        log_event(
            f"STATE AFTER STEP {step}",
            format_state(state),
        )

        messages.append(
            {
                "role": "assistant",
                "content": llm_output,
            }
        )

        if not state.remaining_tasks:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All planned tasks have been completed.\n\n"
                        "Current agent state:\n"
                        + format_state(state)
                        + "\n\nReturn the final answer using the required JSON format."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The tool has been executed.\n\n"
                        "Current agent state:\n"
                        + format_state(state)
                        + "\n\nDecide the next action."
                    ),
                }
            )

    raise RuntimeError(
        f"Agent did not finish after {max_steps} steps."
    )


if __name__ == "__main__":
    main()