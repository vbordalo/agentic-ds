from ollama import chat


MODEL = "qwen2.5:1.5b"


def ns_to_seconds(value: int) -> float:
    return value / 1_000_000_000


def main() -> None:
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise technical assistant specialized "
                    "in data science."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Explain in three sentences what exploratory data "
                    "analysis is and why it is useful."
                ),
            },
        ],
    )

    print("\n--- RESPONSE ---\n")
    print(response.message.content)

    print("\n--- METRICS ---\n")

    total_duration = ns_to_seconds(response.total_duration or 0)
    load_duration = ns_to_seconds(response.load_duration or 0)
    prompt_eval_duration = ns_to_seconds(response.prompt_eval_duration or 0)
    eval_duration = ns_to_seconds(response.eval_duration or 0)

    print(f"Model:                {response.model}")
    print(f"Total duration:       {total_duration:.3f} s")
    print(f"Model load duration:  {load_duration:.3f} s")
    print(f"Prompt eval duration: {prompt_eval_duration:.3f} s")
    print(f"Generation duration:  {eval_duration:.3f} s")

    print(f"Prompt tokens:        {response.prompt_eval_count}")
    print(f"Generated tokens:     {response.eval_count}")

    if response.eval_count and eval_duration > 0:
        tokens_per_second = response.eval_count / eval_duration
        print(f"Generation speed:     {tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    main()