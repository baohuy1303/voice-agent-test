from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import json

llm = ChatOpenAI(model="gpt-5-nano")

system_prompt = """
You are a QA engineer specializing in voice agent testing. Given a description of a voice agent, generate exactly 5 test cases in valid JSON.

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

class PipelineState(TypedDict):
    agent_description: str
    test_cases: list
    issues: list
    final_cases: list
    iterations: int

def generate_test_cases(state: PipelineState):
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["agent_description"]}
    ])
    iterations = state.get("iterations", 0) + 1
    return {
        "test_cases": json.loads(response.content),
        "iterations": iterations
    }

def evaluate_test_cases(state: PipelineState):
    json_format = """
    {
        "issues": [
            {
                "test_case_id": 1,
                "reason": "..."
            }
        ]
    }"""
    prompt = f"""Evaluate these test cases for the agent: {state['agent_description']}
    
    Test cases: {json.dumps(state['test_cases'], indent=2)}

    Flag any that are too unrealistic.
    Return JSON only: {json_format}"""

    response = llm.invoke(prompt)
    return {"issues": json.loads(response.content)["issues"]}

def finalize_test_cases(state: PipelineState):
    flagged_ids = []
    for issue in state["issues"]:
        flagged_ids.append(issue["test_case_id"])
    final_cases = []
    for tc in state["test_cases"]:
        if tc["id"] not in flagged_ids:
            final_cases.append(tc)
    return {"final_cases": final_cases}

def should_regenerate(state: PipelineState):
    if len(state["final_cases"]) < 3 and state["iterations"] < 3:
        return "regenerate"
    return "done"


builder = StateGraph(PipelineState)

builder.add_node("generate", generate_test_cases)
builder.add_node("evaluate", evaluate_test_cases)
builder.add_node("finalize", finalize_test_cases)

builder.set_entry_point("generate")
builder.add_edge("generate", "evaluate")
builder.add_edge("evaluate", "finalize")
builder.add_conditional_edges("finalize", should_regenerate, {
    "regenerate": "generate",
    "done": END
})

graph = builder.compile()