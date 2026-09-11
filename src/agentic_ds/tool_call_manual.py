import json

from ollama import chat


MODEL = "qwen2.5:1.5b"


def add_numbers(a: float, b: float) -> float:
    return a + b


TOOLS = {
    "add_numbers": add_numbers,
}


SYSTEM_PROMPT = """
You are an assistant that can use tools.

Available tools:

1. add_numbers
   Description: Add two numbers.
   Arguments:
   - a: float
   - b: float

When a tool is needed, respond ONLY with valid JSON in this format:

{
  "tool": "add_numbers",
  "arguments": {
    "a": 1.0,
    "b": 2.0
  }
}

Do not calculate the result yourself when the tool should be used.
"""


def main() -> None:
    user_message = "What is 17.5 + 24.3?"

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

    tool_name = tool_request["tool"]
    arguments = tool_request["arguments"]

    tool_function = TOOLS[tool_name]

    tool_result = tool_function(**arguments)

    print("\n--- PYTHON TOOL RESULT ---\n")
    print(tool_result)

    final_response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. "
                    "Use the tool result provided to answer the user."
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
                "content": f"Tool result: {tool_result}",
            },
        ],
    )

    print("\n--- FINAL ANSWER ---\n")
    print(final_response.message.content)


if __name__ == "__main__":
    main()