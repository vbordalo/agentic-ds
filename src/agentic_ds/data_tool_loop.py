import json

from ollama import chat

from data_tools import (
    inspect_dataframe,
    inspect_missing_values,
    load_dataset,
)


MODEL = "qwen2.5:1.5b"

TOOLS = {
    "inspect_dataframe": inspect_dataframe,
    "inspect_missing_values": inspect_missing_values,
}


SYSTEM_PROMPT = """
You are a data science assistant with access to tools.

Available tools:

1. inspect_dataframe
   Description:
   Inspect the structure of the currently loaded pandas DataFrame.

   Returns:
   - number of rows
   - number of columns
   - column names
   - counts of pandas data types
   - approximate memory usage

2. inspect_missing_values
   Description:
   Inspect missing values in the currently loaded pandas DataFrame.

   Returns:
   - columns containing missing values
   - missing count for each affected column
   - total number of missing values

You may use more than one tool if necessary.

If you need to use a tool, respond ONLY with valid JSON:

{
  "type": "tool_call",
  "tool": "<tool_name>",
  "arguments": {}
}

If you already have enough information to answer ALL parts of the
user's request, respond ONLY with valid JSON:

{
  "type": "final_answer",
  "answer": "<your answer>"
}

Important rules:

- Base conclusions only on information explicitly available from tool results.
- Do not assume facts that have not been observed.
- Before giving a final answer, check whether every part of the user's request has been answered.
- If relevant information is still missing, use another appropriate tool.
- Do not invent information about the dataset.
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
    print()

    user_message = input("User: ")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    max_steps = 5

    for step in range(1, max_steps + 1):
        response = chat(
            model=MODEL,
            messages=messages,
        )

        llm_output = response.message.content

        print(f"\n--- LLM DECISION {step} ---\n")
        print(llm_output)

        try:
            decision = json.loads(llm_output)
        except json.JSONDecodeError:
            print("\n--- FINAL ANSWER ---\n")
            print(llm_output)
            return

        if decision["type"] == "final_answer":
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

        tool_result = TOOLS[tool_name](df)

        print(f"\n--- TOOL RESULT {step}: {tool_name} ---\n")
        print(json.dumps(tool_result, indent=2))

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
                    f"Tool result for {tool_name}:\n"
                    + json.dumps(tool_result)
                ),
            }
        )

    raise RuntimeError(
        f"Agent did not finish after {max_steps} steps."
    )


if __name__ == "__main__":
    main()