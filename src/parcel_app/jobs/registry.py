"""The one list of data sources and pipelines that Parcel Studio exposes.

To add a data source (for example HUB) or a pipeline (for example ML),
create its package under ``parcel_app.sources`` or ``parcel_app.pipelines`` and add
it to ``PACKAGES``. Each package's ``__init__`` declares:

- ``JOBS``: a tuple of :class:`JobSpec` for its background work.
- ``ROUTERS``: module paths whose ``router`` is mounted under ``/api``, in order.

Handlers are given as ``"module:function"`` strings, so the API process never
imports heavy pipeline dependencies; the worker child imports them on demand.
See ``docs/web_app/EXTENDING.md``.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module

PACKAGES = (
    "parcel_app.sources.openeo",
    "parcel_app.sources.hub",
    "parcel_app.pipelines.uploads",
    "parcel_app.pipelines.parcel_stats",
    "parcel_app.pipelines.eda",
    "parcel_app.pipelines.ml",
)
LANES = ("run", "small")


@dataclass(frozen=True)
class JobSpec:
    """Describe one queued job kind.

    Attributes:
        kind: Value stored in ``web_jobs.kind``.
        lane: ``"run"`` for long analyses or ``"small"`` for quick metadata work.
        handler: ``"module:function"`` executed with the claimed job in the worker child.
        failure_message: Fixed, browser-safe message stored when the handler fails.
        error_message: Optional message used instead when the exception is known;
            ``{error_type}`` is replaced by the exception class name.
        on_failure: Optional ``"module:function"`` called as ``f(job, message, conn)``
            to mark the job's own record failed in the same transaction.
        interrupt_sql: Statements run by ``reconcile`` for abandoned jobs. Each takes
            one parameter, the list of job kinds being reconciled.
        stoppable: Whether the user can stop a running job of this kind.
    """

    kind: str
    lane: str
    handler: str
    failure_message: str
    error_message: str | None = None
    on_failure: str | None = None
    interrupt_sql: tuple[str, ...] = ()
    stoppable: bool = False


def resolve(target: str):
    """Import a ``"module:attribute"`` reference."""
    module, _, attribute = target.partition(":")
    return getattr(import_module(module), attribute)


@lru_cache(maxsize=1)
def job_specs() -> dict[str, JobSpec]:
    """Return every registered job kind, rejecting duplicates and unknown lanes."""
    specs = {}
    for package in PACKAGES:
        for spec in import_module(package).JOBS:
            if spec.kind in specs:
                raise ValueError(f"Error: Job kind {spec.kind!r} is registered twice.")
            if spec.lane not in LANES:
                raise ValueError(f"Error: Job kind {spec.kind!r} has unknown lane {spec.lane!r}.")
            specs[spec.kind] = spec
    return specs


def lane_kinds(lane: str = "all") -> list[str]:
    """Return the job kinds one worker lane claims; ``"all"`` claims every lane."""
    return [spec.kind for spec in job_specs().values() if lane == "all" or spec.lane == lane]


def router_modules() -> list[str]:
    """Return API router module paths in mounting order."""
    return [name for package in PACKAGES for name in import_module(package).ROUTERS]
