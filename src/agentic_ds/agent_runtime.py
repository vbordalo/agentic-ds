import inspect
import json
from datetime import datetime
from pathlib import Path

import click
from agent_state import AgentState, format_state
from data_tools import (describe_numeric_columns, inspect_dataframe,
                        inspect_missing_values, load_dataset)
from ollama import chat
from planner import create_plan
from synthesizer import synthesize_final_answer

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"agent_run_{RUN_ID}.log"


def log_event(title: str, content: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n--- {title} ---\n")
        f.write(content)
        f.write("\n")



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

If you need to use a tool, respond ONLY with valid JSON
using exactly this structure:

{
  "type": "tool_call",
  "tool": "<tool_name>",
  "arguments": {},
  "tasks": [
    "<exact planned task being addressed>"
  ]
}

When calling a tool, include a non-empty "tasks" list containing
the exact planned task or tasks that the call is intended to address.

The "tasks" field MUST be a JSON list.

Every item in "tasks" MUST exactly match one of the strings currently
listed in remaining_tasks.

Important rules:

- Base conclusions only on information explicitly available from tool results.
- Do not assume facts that have not been observed.
- If relevant information is still missing, use another appropriate tool.
- Do not invent information about the dataset.

The current agent state will be provided after each tool execution.

Use the state to determine:
- what information has already been obtained;
- which tools have already been executed;
- whether additional information is needed.

Do not repeat a tool unless there is a clear reason to do so.
"""


@click.command()
@click.option(
    "--model",
    default="qwen2.5:3b-instruct-q4_K_M",
    show_default=True,
    help="Ollama model used by the agent.",
)
@click.option(
    "--temperature",
    default=0.0,
    type=float,
    show_default=True,
    help="Sampling temperature.",
)
@click.option(
    "--max-steps",
    default=5,
    type=int,
    show_default=True,
    help="Maximum number of executor steps.",
)
def main(
    model: str,
    temperature: float,
    max_steps: int,
) -> None:

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

    log_event(
        "RUN CONFIG",
        json.dumps(
            {
                "model": model,
                "max_steps": max_steps,
                "temperature": temperature,
            },
            indent=2,
        ),
    )

    log_event("USER REQUEST", user_message)

    planned_tasks, planner_output = create_plan(user_message, model, temperature)

    log_event("PLANNER RAW OUTPUT", planner_output)

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
    for step in range(1, max_steps + 1):
        response = chat(
            model=model,
            messages=messages,
            format="json",
            options={"temperature": temperature},
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

        if not state.remaining_tasks:
            final_answer = synthesize_final_answer(state, model, temperature)

            log_event(
                "FINAL ANSWER",
                final_answer,
            )

            print("\n--- FINAL ANSWER ---\n")
            print(final_answer)
            return


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