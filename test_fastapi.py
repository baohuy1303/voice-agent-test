from main import save_and_print_pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from main import generate_test_cases as generate_test_cases_main
from pipeline import graph
from main import save_and_print

app = FastAPI()

class AgentDescription(BaseModel):
    agent_description: str

@app.get('/')
def get_test_cases():
    return {'message': 'Hello World'}

@app.post('/generate-test-cases')
def generate_test_cases(agent_description: AgentDescription):
    res = {'test_cases': generate_test_cases_main(agent_description.agent_description)}
    save_and_print(res, 'test_cases.json')
    return res

class ValidateAgent(BaseModel):
    name: str
    capabilities: list[str]


@app.post('/validate-agent')
def validate_agent(body: ValidateAgent):
    capabilities = body.capabilities
    if len(capabilities) >= 1:
        return {"valid": True, "count": len(capabilities)}
    else:
        return {"valid": False, "count": 0}

@app.post('/run-pipeline')
def run_pipeline(agent_description: AgentDescription):
    res = graph.invoke({"agent_description": agent_description.agent_description})
    save_and_print_pipeline(res['final_cases'], 'test_cases.json')
    return {
        "test_cases": res['final_cases'],
        "issues": res['issues'],
        "iterations": res['iterations']
    }