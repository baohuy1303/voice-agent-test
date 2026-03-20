"""
Voice Agent Test Case Generator
Generates 5 structured test cases for an AI voice agent using OpenAI.
"""

import json
import os
import re
import sys
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# --- Configuration ---
AGENT_DESCRIPTION = """
A dental clinic AI voice receptionist that handles:
- Appointment scheduling, rescheduling, and cancellations
- Basic FAQ (hours, location, accepted insurance, services offered)
- Routing urgent dental emergencies to a human receptionist
- Collecting patient information for new patients
"""

SYSTEM_PROMPT = """You are a QA engineer specializing in voice agent testing. Given a description of a voice agent, generate exactly 5 test cases in valid JSON.

Output ONLY a raw JSON array — no markdown, no code fences, no commentary, no extra text before or after.

Each element in the array must follow this exact schema:
{
  "id": <integer, 1-5>,
  "persona": <string — who the caller is, their age, context, and any relevant traits>,
  "scenario": <string — the specific situation or problem they are calling about>,
  "conversation": [
    {
      "question": <string — what the caller says>,
      "goal": <string — a brief, high-level description of what the agent's response should accomplish for this turn>
    }
  ]
}

Rules:
- The conversation array must have at least 3 turns.
- Each turn's "goal" is a single sentence describing the intent and key outcome of a good agent response (e.g. "Acknowledge the pain, express urgency, and offer to transfer to a human receptionist immediately"). It is not a scripted answer — it is an evaluation checkpoint.
- Personas and scenarios must be diverse across the 5 test cases.
- Cover both happy paths and edge/failure cases (e.g. urgent situation, confused caller, out-of-scope request).
- Do not include any text outside the JSON array.
"""

class ConversationTurn(BaseModel):
    question: str
    goal: str

class TestCase(BaseModel):
    id: int = Field(..., ge=1, le=5)
    persona: str
    scenario: str
    conversation: List[ConversationTurn]

class TestCases(BaseModel):
    test_cases: List[TestCase]


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if the model wraps output in them."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_test_cases(agent_description: str) -> list[dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Generate 5 test cases for the following voice agent:\n\n{agent_description.strip()}",
            },
        ],
        response_format=TestCases
    )

    raw = response.choices[0].message.content
    #print(raw)
    cleaned = strip_markdown_fences(raw)

    try:
        test_cases = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response as JSON: {e}", file=sys.stderr)
        print(f"[RAW OUTPUT]\n{cleaned}", file=sys.stderr)
        sys.exit(1)

    """if not isinstance(test_cases, list) or len(test_cases) != 5:
        print(
            f"[ERROR] Expected a list of 5 test cases, got: {type(test_cases).__name__} with {len(test_cases) if isinstance(test_cases, list) else 'N/A'} items",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate required keys per test case
    required_keys = {"id", "persona", "scenario", "conversation"}
    for i, tc in enumerate(test_cases):
        missing = required_keys - tc.keys()
        if missing:
            print(f"[ERROR] Test case {i+1} is missing keys: {missing}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(tc["conversation"], list) or len(tc["conversation"]) < 3:
            print(
                f"[ERROR] Test case {i+1} 'conversation' must have at least 3 turns.",
                file=sys.stderr,
            )
            sys.exit(1)
        for j, turn in enumerate(tc["conversation"]):
            if "question" not in turn or "goal" not in turn:
                print(
                    f"[ERROR] Test case {i+1}, turn {j+1} is missing 'question' or 'goal'.",
                    file=sys.stderr,
                )
                sys.exit(1)"""

    return test_cases


def save_and_print(test_cases: list[dict], output_file: str = "test_cases.json") -> None:
    output = json.dumps(test_cases, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    num_of_test_cases = 0
    for test_case in test_cases["test_cases"]:
        num_of_test_cases += 1
    print(f"\n[OK] Saved {num_of_test_cases} test cases to '{output_file}'", file=sys.stderr)


if __name__ == "__main__":
    test_cases = generate_test_cases(AGENT_DESCRIPTION)
    save_and_print(test_cases)
