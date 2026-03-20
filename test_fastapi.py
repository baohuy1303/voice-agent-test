from main import save_and_print
from fastapi import FastAPI
from pydantic import BaseModel
from main import generate_test_cases as generate_test_cases_main

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