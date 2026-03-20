def check_valid(capabilities: list[str]) -> dict:
    if len(capabilities) >= 1:
        return {"valid": True, "count": len(capabilities)}
    else:
        return {"valid": False, "count": 0}

def test_with_capabilities():
    res = check_valid(["scheduling", "FAQ"])
    assert res == {"valid": True, "count": 2}

def test_empty_capabilities():
    res = check_valid([])
    assert res == {"valid": False, "count": 0}