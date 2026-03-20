# voice-agent-test

Just a repo for me to mess around with Python, Pytest, LangChain, LangGraph, and testing stuff for learning purposes. Feel free to take a look.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the basic generator:
```bash
python main.py
```

Run the FastAPI server and call the LangGraph pipeline:
```bash
uvicorn test_fastapi:app --reload
```

Call your port (default: http://localhost:8000/run-pipeline) with a POST request with the JSON body:
```json
{
    "agent_description": "<agent description>"
}
```

## Testing

```bash
pytest
```
