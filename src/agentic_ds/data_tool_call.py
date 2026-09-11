import json

from ollama import chat

from data_tools import inspect_dataframe, load_dataset, inspect_missing_values


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

When a tool is needed, respond ONLY with valid JSON in this format:

{
  "tool": "<tool_name>",
  "arguments": {}
}

Do not invent information about the dataset.
Use the appropriate tool whenever the requested information depends on the data.
"""

def main() -> None:
    df = load_dataset()

    print("\nExamples of questions you can ask:")
    print("- How many rows and columns are in the dataset?")
    print("- Tell me the dimensions and data types of this dataset.")
    print("- Give me an overview of the dataset structure.")
    print("- Are there missing values?")
    print("- Which columns contain missing values?")
    print("- Which variables have incomplete data?")
    print()
    user_message = input("User: ")


    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    llm_output = response.message.content

    print("\n--- LLM TOOL REQUEST ---\n")
    print(llm_output)

    tool_request = json.loads(llm_output)

    # Executor: Validate the tool request and execute the corresponding function
    tool_name = tool_request["tool"]

    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_result = TOOLS[tool_name](df)

    print("\n--- PYTHON TOOL RESULT ---\n")
    print(json.dumps(tool_result, indent=2))

    final_response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise data science assistant. "
                    "Answer strictly from the tool result provided."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": llm_output,
            },
            {
                "role": "user",
                "content": (
                    "Tool result:\n"
                    + json.dumps(tool_result)
                ),
            },
        ],
    )

    print("\n--- FINAL ANSWER ---\n")
    print(final_response.message.content)


if __name__ == "__main__":
    main()