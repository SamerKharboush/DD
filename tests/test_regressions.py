"""Regression tests for the tool registry and tool-call parser.

These cover defects that shipped in the tree and were fixed: all four tool
modules failed to import because `ToolRegistry.register` could not serve as a
decorator factory, and the tool-call regex could not match any namespaced tool
name. Both were silent at runtime — the tools simply did not exist.
"""

import pytest


# ---------------------------------------------------------------------------
# Registry: decorator-factory registration
# ---------------------------------------------------------------------------

def test_four_tool_modules_import():
    """All tool modules must import; previously all four raised TypeError."""
    import importlib

    for mod in (
        "ct.tools.phase1_tools",
        "ct.tools.phase2_tools",
        "ct.tools.phase3_tools",
        "ct.tools.structure_io_tools",
    ):
        importlib.import_module(mod)


def test_registry_registers_forty_tools():
    """The 40 decorator-registered tools must actually be present."""
    from ct.tools.registry import registry

    names = registry.list_tools()
    assert len(names) >= 40, f"only {len(names)} tools registered"


def test_register_supports_decorator_form():
    """`@registry.register(name=..., description=...)` must work."""
    from ct.tools.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.register(name="x.y", description="d", category="c")
    def my_tool(a: int = 1):
        return {"a": a}

    assert "x.y" in reg._tools
    # The decorator must return the function unchanged so the module-level
    # name stays bound to the function itself.
    assert my_tool(5) == {"a": 5}


def test_register_supports_direct_form():
    """`registry.register("name", tool)` must still work."""
    from ct.tools.registry import BaseTool, ToolRegistry

    class T(BaseTool):
        name = "direct.tool"

        def run(self, **kwargs):
            return {"ok": True}

    reg = ToolRegistry()
    reg.register("direct.tool", T())
    assert reg.get_tool("direct.tool") is not None


def test_decorator_preserves_function_identity():
    """The registered object must be the function, not a wrapper."""
    from ct.tools.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.register(name="z", description="", category="")
    def fn():
        return 1

    assert reg._tools["z"] is fn


# ---------------------------------------------------------------------------
# Placeholder tools must not fabricate data
# ---------------------------------------------------------------------------

def test_unavailable_tool_raises_instead_of_returning_constants():
    """Placeholder tools must raise, never return invented measurements."""
    from ct.tools.registry import UnavailableTool

    t = UnavailableTool()
    t.name = "boltz2.predict_affinity"
    with pytest.raises(NotImplementedError):
        t.run(protein_sequence="MKV", ligand_smiles="CCO")


def test_real_tools_are_not_shadowed_by_mocks():
    """The real boltz2 tool must not be replaced by a hardcoded stub."""
    from ct.tools.registry import UnavailableTool, registry

    registry._lazy_init()
    names = registry.list_tools()
    for n in names:
        if n == "boltz2.predict_affinity":
            assert not isinstance(registry.get_tool(n), UnavailableTool), (
                "real boltz2.predict_affinity was overwritten by a stub"
            )


def test_gpu_service_reports_placeholder_honestly():
    """is_available() must be False by default so the 503 guard can fire."""
    from ct.gpu.boltz2_service import Boltz2Service

    svc = Boltz2Service()
    assert svc.is_available() is False
    assert svc.load_model() is False
    status = svc.get_status()
    assert status["is_real_inference"] is False
    assert status["placeholder_output"] is True


def test_gpu_prediction_marks_itself_as_not_real():
    """Placeholder predictions must carry is_real_result=False."""
    from ct.gpu.boltz2_service import Boltz2Service

    svc = Boltz2Service()
    p = svc.predict_structure("MKV")
    assert p.is_real_result is False
    a = svc.predict_affinity("MKV", "CCO")
    assert a.is_real_result is False


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    from ct.agent.runner import AgentRunner

    return AgentRunner.__new__(AgentRunner)._extract_tool_calls


@pytest.mark.parametrize("name", [
    "admet.predict",
    "boltz2.predict_affinity",
    "knowledge.search_entities",
    "structure.parse_pdb",
    "generative.design_binder",
])
def test_namespaced_tool_names_are_extracted(extractor, name):
    """`\\w+` never matched a dot, so no real tool could ever be parsed."""
    calls = extractor(f'TOOL: {name}("x")')
    assert len(calls) == 1
    assert calls[0]["tool"] == name


def test_bare_kwargs_are_parsed_not_raw(extractor):
    """`smiles="CCO"` must not degrade to {"raw": ...}."""
    calls = extractor('TOOL: admet.predict(smiles="CCO", limit=20)')
    assert calls[0]["params"] == {"smiles": "CCO", "limit": 20}


def test_json_object_arguments(extractor):
    calls = extractor('TOOL: f({"a": 1, "b": "x"})')
    assert calls[0]["params"] == {"a": 1, "b": "x"}


def test_nested_parens_not_truncated(extractor):
    """Non-greedy `(.*?)` used to stop at the first closing paren."""
    calls = extractor('TOOL: f(a=1, g(2, 3), b="x")')
    assert len(calls) == 1
    p = calls[0]["params"]
    assert p["a"] == 1
    assert p["b"] == "x"
    assert "g(2, 3)" in p.values()


def test_multiple_calls_in_one_response(extractor):
    calls = extractor(
        'TOOL: admet.predict("C") then TOOL: boltz2.predict_affinity("P","L")'
    )
    assert [c["tool"] for c in calls] == [
        "admet.predict", "boltz2.predict_affinity",
    ]


def test_bare_tool_marker_is_flagged_not_dropped(extractor):
    calls = extractor("TOOL: admet.predict")
    assert len(calls) == 1
    assert "_parse_error" in calls[0]


def test_no_marker_returns_empty(extractor):
    assert extractor("just some prose") == []
    assert extractor("") == []


def test_unbalanced_parens_do_not_crash(extractor):
    assert isinstance(extractor('TOOL: f(a="unclosed'), list)


def test_separator_split_respects_nesting_and_quotes():
    from ct.agent.runner import _split_top_level

    assert _split_top_level("a=1, g(2, 3), b=x") == ["a=1", "g(2, 3)", "b=x"]
    assert _split_top_level('s="a,b"') == ['s="a,b"']


# ---------------------------------------------------------------------------
# Auth: default-deny
# ---------------------------------------------------------------------------

def test_protected_endpoints_require_auth():
    """Endpoints that spend vendor credit must reject anonymous callers."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ct.api.main import app

    with TestClient(app) as c:
        assert c.post("/api/v1/query", json={"query": "x"}).status_code == 401
        assert c.post("/api/v1/multi-agent", json={"query": "x"}).status_code == 401
        assert c.post("/api/v1/dmta", json={"query": "x"}).status_code == 401
        assert c.get("/api/v1/models").status_code == 401
        assert c.get("/metrics").status_code == 401
        assert c.get("/api/v1/health/detailed").status_code == 401
        assert c.post("/api/v1/rlef/train", json={}).status_code == 401
        assert c.post("/api/v1/api-keys", json={"name": "k"}).status_code == 401


def test_public_endpoints_stay_open():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ct.api.main import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/").status_code == 200


def test_bad_token_is_rejected():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ct.api.main import app

    with TestClient(app) as c:
        r = c.get("/api/v1/models", headers={"Authorization": "Bearer bad.token.x"})
        assert r.status_code == 401


def test_cors_does_not_reflect_hostile_origin():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ct.api.main import app

    with TestClient(app) as c:
        r = c.get("/health", headers={"Origin": "https://evil.example.com"})
        assert r.headers.get("access-control-allow-origin") is None
        assert r.headers.get("access-control-allow-credentials") is None


# ---------------------------------------------------------------------------
# run_query must not let callers choose the model
# ---------------------------------------------------------------------------

def test_run_query_context_cannot_set_model():
    """`{"context": {"model": ...}}` must not reach run_query's model param."""
    from ct.agent import runner as R

    captured = {}

    class Fake:
        def __init__(self, model=None):
            captured["model"] = model

        def run(self, query, context=None):
            captured["context"] = context
            return {}

    orig = R.AgentRunner
    R.AgentRunner = Fake
    try:
        R.run_query("hi", context={"model": "attacker-model"})
    finally:
        R.AgentRunner = orig

    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["context"] == {"model": "attacker-model"}


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def test_jwt_round_trip():
    from ct.security.auth import create_access_token, verify_token

    secret = "unit-test-secret-value-not-a-real-key"
    tok = create_access_token("u1", "alice", role="admin", secret_key=secret)
    td = verify_token(tok, secret_key=secret)
    assert td is not None
    assert td.user_id == "u1"
    assert td.role == "admin"


def test_jwt_rejects_alg_none():
    import base64
    import json
    import time

    from ct.security.auth import verify_token

    h = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    p = base64.urlsafe_b64encode(json.dumps({
        "user_id": "u1", "username": "a", "role": "admin",
        "iat": int(time.time()), "exp": int(time.time()) + 9999,
    }).encode()).decode().rstrip("=")

    assert verify_token(f"{h}.{p}.", secret_key="s") is None


def test_jwt_rejects_wrong_secret_and_expiry():
    from ct.security.auth import create_access_token, verify_token

    secret = "unit-test-secret-value-not-a-real-key"
    tok = create_access_token("u1", "alice", secret_key=secret)
    assert verify_token(tok, secret_key="other") is None

    expired = create_access_token("u1", "alice", secret_key=secret, expires_hours=-1)
    assert verify_token(expired, secret_key=secret) is None