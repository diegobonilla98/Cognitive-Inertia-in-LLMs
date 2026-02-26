import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from cognitive_inertia.paths import SYSTEM_PROMPT_PATH

load_dotenv()

SMART_MODEL = "gpt-5.2-2025-12-11"
STUPID_MODEL = "gpt-4o-mini-2024-07-18"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _read_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8").strip()


DEVELOPER_INSTRUCTIONS = _read_system_prompt()


def call_gpt52_smart_messages(
    messages: list[dict[str, str]], developer_instructions: str = DEVELOPER_INSTRUCTIONS
) -> str:
    response = client.responses.create(
        model=SMART_MODEL,
        instructions=developer_instructions,
        input=messages,
        reasoning={"effort": "medium"},
        text={"format": {"type": "text"}, "verbosity": "medium"},
    )
    return response.output_text


def call_gpt52_smart(user_text: str, developer_instructions: str = DEVELOPER_INSTRUCTIONS) -> str:
    return call_gpt52_smart_messages(
        [{"role": "user", "content": user_text}],
        developer_instructions=developer_instructions,
    )


def call_4o_mini_stupid_messages(
    messages: list[dict[str, str]], developer_instructions: str = DEVELOPER_INSTRUCTIONS
) -> str:
    response = client.responses.create(
        model=STUPID_MODEL,
        instructions=developer_instructions,
        input=messages,
        temperature=1,
        top_p=1,
        max_output_tokens=2048,
        text={"format": {"type": "text"}},
    )
    return response.output_text


def call_4o_mini_stupid(user_text: str, developer_instructions: str = DEVELOPER_INSTRUCTIONS) -> str:
    return call_4o_mini_stupid_messages(
        [{"role": "user", "content": user_text}],
        developer_instructions=developer_instructions,
    )
