import json
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

load_dotenv()

# ---------------------------------------------------------------------------
# Models from gen_scenario.py
# ---------------------------------------------------------------------------

class StateConfig(BaseModel):
    name: str
    prompt: str
    modelName: str
    transitions: List[str]
    initialMessage: Optional[str] = None


class AgentConfig(BaseModel):
    actions: List[str]
    initialState: StateConfig
    additionalStates: List[StateConfig]


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


# ---------------------------------------------------------------------------
# Task-specific Models
# ---------------------------------------------------------------------------

class Transcript(BaseModel):
    ai: List[str]
    human: List[str]


class ValidationOutput(BaseModel):
    human_strings: List[str]
    ai_strings: List[str]
    summary: str
    score: int
    comments: str

json_format = """
{
  "ai": [
    "Hello, how can I help you today?",
    "Thank you for providing that information. I'll transfer you to our billing department now."
  ],
  "human": [
    "Hi, I have a question about a bill I received.",
    "Yes, I'm a bit confused about the charges on my latest statement."
  ]
}
"""


# ---------------------------------------------------------------------------
# Transcription & Validation Functions
# ---------------------------------------------------------------------------

def generate_mock_transcript(scenario: Scenario, config: AgentConfig, client: OpenAI) -> Transcript:
    """
    Generates a mock transcript following the back-and-forth pattern.
    AI starts first.
    """
    system_prompt = (
        "You are an AI simulator. Your task is to generate a realistic transcript "
        "of a conversation between a clinic's AI receptionist and a customer based on a provided scenario."
    )

    all_states = [config.initialState] + config.additionalStates
    state_summary = "\n".join(
        f"- {s.name}: {s.prompt}" for s in all_states
    )

    user_prompt = f"""
Scenario Name: {scenario.scenarioName}
Scenario Description: {scenario.scenarioDescription}
Customer Info:
- Name: {scenario.name}
- DOB: {scenario.dob}
- Phone: {scenario.phone}
- Email: {scenario.email}
- Gender: {scenario.gender}
- Insurance: {scenario.insurance}

AI Agent Configuration:
{state_summary}

Initial AI Message: {config.initialState.initialMessage or "Hello, how can I help you today?"}

Generate a transcript of the conversation. 
Follow these rules:
1. The AI starts first with the initial message.
2. The customer responds based on the scenario.
3. The conversation continues back and forth until the goal is reached or all transition/states has been covered.
4. Output the result in JSON format with two keys: "ai" (array of strings) and "human" (array of strings).
5. The length of the arrays should be nearly equal, representing the back-and-forth nature.
6. The first element of "ai" MUST be the initial AI message.

Return only valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = response.choices[0].message.content
        return Transcript.model_validate_json(raw_content)
    except (OpenAIError, ValidationError) as e:
        print(f"Error generating transcript for {scenario.scenarioName}: {e}")
        return Transcript(ai=[], human=[])


def validate_transcript(transcript: Transcript, scenario: Scenario, client: OpenAI) -> ValidationOutput:
    """
    Validates the transcript against scenario description and criteria.
    """
    system_prompt = (
        "You are a QA analyst. Your goal is to evaluate a transcript of an AI agent "
        "interaction against a specific scenario and success criteria."
    )

    json_format = """
    {
    "summary": "...",
    "score": 10,
    "comments": "..."
    }
    """


    # Reconstruct the conversation for the LLM to read
    conversation_text = ""
    max_len = max(len(transcript.ai), len(transcript.human))
    for i in range(max_len):
        if i < len(transcript.ai):
            conversation_text += f"AI: {transcript.ai[i]}\n"
        if i < len(transcript.human):
            conversation_text += f"Human: {transcript.human[i]}\n"

    user_prompt = f"""
Scenario Description: {scenario.scenarioDescription}
Criteria: {scenario.criteria}

Transcript:
{conversation_text}

Evaluate the transcript:
1. Provide a concise summary of the conversation.
2. Rate the AI agent's performance on a scale of 1-10 (10 being perfect adherence to criteria and smooth interaction).
3. Provide detailed comments on why this score was given.

Output JSON:
{json_format}
Return only valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = response.choices[0].message.content
        eval_data = json.loads(raw_content)
        
        return ValidationOutput(
            human_strings=transcript.human,
            ai_strings=transcript.ai,
            summary=eval_data.get("summary", ""),
            score=eval_data.get("score", 0),
            comments=eval_data.get("comments", "")
        )
    except (OpenAIError, ValidationError, json.JSONDecodeError) as e:
        print(f"Error validating transcript for {scenario.scenarioName}: {e}")
        return ValidationOutput(
            human_strings=transcript.human,
            ai_strings=transcript.ai,
            summary="Error in validation",
            score=0,
            comments=str(e)
        )


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)

    # 1. Load Scenarios
    try:
        with open("scenarios.json", "r", encoding="utf-8") as f:
            scenarios_raw = json.load(f)
            # Validate with Pydantic
            scenarios = [Scenario.model_validate(s) for s in scenarios_raw]
    except (FileNotFoundError, ValidationError, json.JSONDecodeError) as e:
        print(f"Error loading scenarios.json: {e}")
        sys.exit(1)

    # 2. Get Agent Config (using the sample from the conversation/gen_scenario.py)
    # Typically this would be loaded from a file or the gen_scenario script.
    # For now, let's use the sample provided in gen_scenario.py
    sample_config_json = {
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
    
    try:
        config = AgentConfig.model_validate(sample_config_json)
    except ValidationError as e:
        print(f"Error validating agent config: {e}")
        sys.exit(1)

    # 3. Process each scenario
    results = []
    print(f"Starting validation for {len(scenarios)} scenarios...")
    
    for scenario in scenarios:
        print(f"Processing: {scenario.scenarioName}")
        
        # Step A: Generate Mock Transcript
        transcript = generate_mock_transcript(scenario, config, client)
        
        if not transcript.ai or not transcript.human:
            print(f"Skipping {scenario.scenarioName} due to generation error.")
            continue
            
        # Step B: Validate Transcript
        validation_output = validate_transcript(transcript, scenario, client)
        
        results.append(validation_output.model_dump())

    # 4. Save results
    output_file = "validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Finished! Results saved to {output_file}")


if __name__ == "__main__":
    main()
