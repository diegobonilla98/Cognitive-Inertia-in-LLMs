import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SMART_MODEL = "gpt-5.2-2025-12-11"
STUPID_MODEL = "gpt-4o-mini-2024-07-18"

with open("system_prompt.txt", "r") as file:
    DEVELOPER_INSTRUCTIONS = file.read()


def call_gpt52_smart_messages(
    messages: list[dict[str, str]], developer_instructions: str = DEVELOPER_INSTRUCTIONS
) -> str:
    resp = client.responses.create(
        model=SMART_MODEL,
        instructions=developer_instructions,
        input=messages,
        reasoning={"effort": "medium"},
        text={
            "format": {"type": "text"},
            "verbosity": "medium",
        },
    )
    return resp.output_text


def call_gpt52_smart(user_text: str, developer_instructions: str = DEVELOPER_INSTRUCTIONS) -> str:
    return call_gpt52_smart_messages(
        [{"role": "user", "content": user_text}],
        developer_instructions=developer_instructions,
    )


def call_4o_mini_stupid_messages(
    messages: list[dict[str, str]], developer_instructions: str = DEVELOPER_INSTRUCTIONS
) -> str:
    resp = client.responses.create(
        model=STUPID_MODEL,
        instructions=developer_instructions,
        input=messages,
        temperature=1,
        top_p=1,
        max_output_tokens=2048,
        text={"format": {"type": "text"}},
    )
    return resp.output_text


def call_4o_mini_stupid(user_text: str, developer_instructions: str = DEVELOPER_INSTRUCTIONS) -> str:
    return call_4o_mini_stupid_messages(
        [{"role": "user", "content": user_text}],
        developer_instructions=developer_instructions,
    )


if __name__ == "__main__":
    prompt = "Solve: If x^2 = 9, what are the possible values of x?"

    print("=== GPT-5.2 (smart) ===")
    print(call_gpt52_smart(prompt))

    print("\n=== GPT-4o-mini (stupid) ===")
    print(call_4o_mini_stupid(prompt))
