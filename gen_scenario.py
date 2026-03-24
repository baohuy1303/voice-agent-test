import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

load_dotenv()


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class StateConfig(BaseModel):
    name: str
    prompt: str
    modelName: str
    transitions: list[str]
    initialMessage: str | None = None


class AgentConfig(BaseModel):
    actions: list[str]
    initialState: StateConfig
    additionalStates: list[StateConfig]


class InputPayload(BaseModel):
    agentConfig: AgentConfig


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    scenarioName: str
    scenarioDescription: str
    name: str
    dob: str
    phone: str
    email: str
    gender: str
    insurance: str
    criteria: str


class ScenarioList(BaseModel):
    scenarios: list[Scenario]


json_format = """
{
    "scenarioName": "Returning Patient - Basic Appointment Request",
    "scenarioDescription": "John Doe, a returning patient, calls the clinic to set up a follow-up appointment. He provides his phone number, which the system has on file from a previous visit. He expects the agent to correctly recognize him as a returning patient and confirm his existing details. He wants to schedule a general check-up for next week. He also wonders if his insurance is still on file or if he needs to provide updated information.",
    "name": "John Doe",
    "dob": "01/01/1980",
    "phone": "202-360-9536",
    "email": "john.doe@example.com",
    "gender": "Male",
    "insurance": "Aetna",
    "criteria": "The agent smoothly recognizes John as a returning patient using his phone number, verifies personal details, and helps schedule a check-up appointment."
}
"""

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def create_scenarios(
    input_json_str: str,
    num_scenarios: int,
    output_file: str = "scenarios.json",
) -> list[Scenario]:
    # 1. Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    # 2. Parse & validate input
    try:
        payload = InputPayload.model_validate_json(input_json_str)
    except ValidationError as e:
        print("Invalid agentConfig input:")
        print(e)
        sys.exit(1)

    config = payload.agentConfig
    all_states = [config.initialState] + config.additionalStates
    state_summary = "\n".join(
        f"- {s.name}: transitions={s.transitions}" for s in all_states
    )

    # 3. Build prompts
    system_prompt = (
        "You are a QA engineer designing complex behavioral test scenarios for an AI "
        "voice agent. Each scenario must exercise realistic interactions and "
        "cover diverse state transition paths through the agent's workflow."
    )

    user_prompt = f"""
Generate exactly {num_scenarios} unique test scenarios for the following AI agent.

Agent actions: {config.actions}
States and transitions:
{state_summary}

Initial state prompt: {config.initialState.prompt}

Requirements:
- Cover a variety of transition paths (e.g. intial state to other states, additional states to other states, edge cases).
- Each scenario must be realistic and distinct.
- Return a JSON object with a single key "scenarios" containing an array of {num_scenarios} objects.
- Each object must have exactly these fields:
  scenarioName, scenarioDescription, name, dob, phone, email, gender, insurance, criteria

Example of a scenario:
{json_format}

Return only valid JSON, no extra text.
"""

    # 4. Call OpenAI
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

    raw_content = response.choices[0].message.content

    # 5. Validate output
    try:
        result = ScenarioList.model_validate_json(raw_content)
    except ValidationError as e:
        print("OpenAI response did not match expected schema:")
        print(e)
        sys.exit(1)

    # 6. Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in result.scenarios], f, indent=2)

    print(f"Wrote {len(result.scenarios)} scenarios to {output_file}")
    return result.scenarios


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_config = """{
  "agentConfig": {
    "actions": ["find_patient_info", "dial_human_agent"],
    "initialState": {
      "name": "INFORMATION_COLLECTION",
      "prompt": "You are an AI receptionist for a clinic. Your goal is to gather patient information, such as contact details and insurance, or look up existing patient records. If the patient needs to schedule an appointment, transition to SCHEDULING_APPOINTMENT. If the call involves a medical history discussion, transition to HPI_COLLECTION. For all other inquiries, connect to a human agent. Always ask for clarifications, ensure accuracy, and thank the patient before ending the call.",
      "modelName": "gpt-4o",
      "transitions": ["SCHEDULING_APPOINTMENT", "HPI_COLLECTION"],
      "initialMessage": "Hello, thank you for calling the clinic. I'm your AI receptionist. Are you a new or returning patient?"
    },
    "additionalStates": [
      {
        "name": "SCHEDULING_APPOINTMENT",
        "prompt": "You are scheduling an appointment for a clinic. Ask the patient their appointment type and offer available slots (e.g., Tuesdays 4-5 PM, Wednesdays 2-3 PM, Fridays 9-10 AM). Confirm or reschedule as needed. Always thank the patient and end the call politely.",
        "modelName": "gpt-4o-mini",
        "transitions": []
      },
      {
        "name": "HPI_COLLECTION",
        "prompt": "You are collecting medical history for a patient's upcoming doctor visit. Gather information about their symptoms, medical history, medications, and allergies. Ask clear, step-by-step questions to prepare the doctor for the visit. If unsure about anything, connect to a human agent and thank the patient before ending the call.",
        "modelName": "gpt-4o-mini",
        "transitions": []
      }
    ]
  }
}"""

    create_scenarios(sample_config, num_scenarios=3)
