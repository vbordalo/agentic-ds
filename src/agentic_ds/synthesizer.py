from ollama import chat

from agent_state import AgentState, format_state

FINAL_SYNTHESIS_PROMPT = """
You are the final response component of a data science assistant.

Answer the user's original request using ONLY the evidence contained
in the provided agent state.

Rules:
- Do not invent facts or calculations.
- Distinguish counts, means, standard deviations, percentages,
  and missing-value counts.
- Do not describe analyses that were not performed.
- Do not claim that all columns have a property when the evidence
  only identifies some columns.
- Do not mention future work or remaining tasks when there are none.
- Be concise and directly answer the user's request.
"""


def synthesize_final_answer(state: AgentState, model: str, temperature: float) -> str:
    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": FINAL_SYNTHESIS_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Original user request:\n"
                    + state.user_request
                    + "\n\nAgent state and collected evidence:\n"
                    + format_state(state)
                ),
            },
        ],
        options={
            "temperature": temperature,
        },
    )

    final_output = response.message.content

    return final_output
