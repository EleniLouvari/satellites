"""The job registry is the single source for job kinds, lanes, failure handling and routers."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from parcel_app.jobs import dispatch, registry
from parcel_app.jobs.registry import JobSpec


def test_lanes_keep_the_original_job_split():
    """Long analyses and quick metadata work stay in separate worker lanes."""
    assert sorted(registry.lane_kinds("run")) == ["bundle", "eda", "ml", "resume", "run"]
    assert sorted(registry.lane_kinds("small")) == ["inspect", "test_account", "validate"]
    assert sorted(registry.lane_kinds("all")) == sorted(registry.lane_kinds("run") + registry.lane_kinds("small"))


def test_every_handler_hook_and_router_resolves():
    """String references are importable, so a typo fails here rather than in a worker child."""
    for spec in registry.job_specs().values():
        assert callable(registry.resolve(spec.handler))
        if spec.on_failure:
            assert callable(registry.resolve(spec.on_failure))
    for name in registry.router_modules():
        assert hasattr(registry.resolve(f"{name}:router"), "routes")


def test_only_parcel_analyses_are_stoppable():
    """Stop requests apply only to run and resume claims."""
    assert sorted(kind for kind, spec in registry.job_specs().items() if spec.stoppable) == ["resume", "run"]


def test_interrupt_statements_take_the_kind_list_parameter():
    """Reconciliation passes exactly one parameter, the reconciled job kinds."""
    for spec in registry.job_specs().values():
        for sql in spec.interrupt_sql:
            assert sql.count("%s") == 1 and "kind=ANY(%s)" in sql


def _register(monkeypatch, *specs):
    """Replace the registered packages with one synthetic package."""
    package = SimpleNamespace(JOBS=specs, ROUTERS=())
    monkeypatch.setattr(registry, "PACKAGES", ("synthetic",))
    monkeypatch.setattr(registry, "import_module", lambda name: package)
    registry.job_specs.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_registry_cache():
    """Never leak a synthetic registry into other tests."""
    registry.job_specs.cache_clear()
    yield
    registry.job_specs.cache_clear()


def test_duplicate_kinds_and_unknown_lanes_are_rejected(monkeypatch):
    """Two packages cannot claim the same kind, and every kind needs a real lane."""
    spec = JobSpec(kind="same", lane="run", handler="x:y", failure_message="Error: x")
    _register(monkeypatch, spec, spec)
    with pytest.raises(ValueError, match="registered twice"):
        registry.job_specs()
    _register(monkeypatch, JobSpec(kind="odd", lane="gpu", handler="x:y", failure_message="Error: x"))
    with pytest.raises(ValueError, match="unknown lane"):
        registry.job_specs()


def test_failure_uses_fixed_message_hook_and_error_type(monkeypatch):
    """A failed job records the fixed or error-type message and calls its own failure hook."""
    hooked, writes = [], []
    spec = JobSpec(
        kind="demo", lane="run", handler="x:y", failure_message="Error: fixed.",
        error_message="Error: failed ({error_type}).", on_failure="hook:mark",
    )
    monkeypatch.setattr(dispatch, "job_specs", lambda: {"demo": spec})
    monkeypatch.setattr(dispatch, "resolve", lambda target: lambda job, message, conn: hooked.append(message))
    monkeypatch.setattr(dispatch, "owned_claim", lambda job, conn: None)
    monkeypatch.setattr(dispatch.db, "connection", lambda: nullcontext(object()))
    monkeypatch.setattr(dispatch.db, "execute", lambda sql, params, **kw: writes.append(params[0]))
    job = {"job_id": "1", "kind": "demo"}
    dispatch.fail(job, KeyError("secret detail"))
    dispatch.fail(job)
    assert hooked == writes == ["Error: failed (KeyError).", "Error: fixed."]
