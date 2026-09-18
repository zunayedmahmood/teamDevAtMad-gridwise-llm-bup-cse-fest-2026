import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable

from app.config import settings
from app.interpreter.prompt import PROMPT_VERSION
from app.models.internal import CanonicalRequest, ValidatedDirective
from app.models.response import DirectiveType


def compute_cache_key(
    canonical: CanonicalRequest,
    model: str | None = None,
    prompt_version: str | None = None,
) -> str:
    prompt_ver = prompt_version if prompt_version is not None else PROMPT_VERSION
    model_id = model if model is not None else settings.openai_model
    battery = canonical.battery
    payload: dict[str, Any] = {
        "prompt_version": prompt_ver,
        "model": model_id,
        "notes": list(canonical.operator_notes),
        "battery_context": {
            "capacity_kwh": float(battery.capacity_kwh),
            "initial_energy_kwh": float(battery.initial_energy_kwh),
            "minimum_energy_kwh": float(battery.minimum_energy_kwh),
            "max_charge_kwh_per_hour": float(battery.max_charge_kwh_per_hour),
            "max_discharge_kwh_per_hour": float(battery.max_discharge_kwh_per_hour),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def serialize_directives(directives: list[ValidatedDirective]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for d in directives:
        result.append(
            {
                "note_index": d.note_index,
                "applies": d.applies,
                "directive_type": d.directive_type.value,
                "hours": list(d.hours) if d.hours is not None else None,
                "factor": d.factor,
                "minimum_energy_kwh": d.minimum_energy_kwh,
                "max_grid_kwh": d.max_grid_kwh,
                "explanation": d.explanation,
            }
        )
    return result


def deserialize_directives(data: list[dict[str, Any]]) -> list[ValidatedDirective]:
    directives: list[ValidatedDirective] = []
    for item in data:
        directives.append(
            ValidatedDirective(
                note_index=int(item["note_index"]),
                applies=bool(item["applies"]),
                directive_type=DirectiveType(item["directive_type"]),
                hours=tuple(item["hours"]) if item.get("hours") is not None else None,
                factor=float(item["factor"]) if item.get("factor") is not None else None,
                minimum_energy_kwh=(
                    float(item["minimum_energy_kwh"])
                    if item.get("minimum_energy_kwh") is not None
                    else None
                ),
                max_grid_kwh=(
                    float(item["max_grid_kwh"])
                    if item.get("max_grid_kwh") is not None
                    else None
                ),
                explanation=str(item["explanation"]),
            )
        )
    return directives


class InterpretationCache:
    def __init__(
        self,
        max_entries: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = (
            OrderedDict()
        )
        self._lock = asyncio.Lock()

    @property
    def max_entries(self) -> int:
        return (
            self._max_entries
            if self._max_entries is not None
            else settings.interpretation_cache_max_entries
        )

    @property
    def ttl_seconds(self) -> float:
        return (
            self._ttl_seconds
            if self._ttl_seconds is not None
            else settings.interpretation_cache_ttl_seconds
        )

    async def get(self, key: str) -> list[ValidatedDirective] | None:
        if not settings.interpretation_cache_enabled:
            return None
        async with self._lock:
            if key not in self._cache:
                return None
            created_at, serialized = self._cache[key]
            if (time.monotonic() - created_at) > self.ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return deserialize_directives(serialized)

    async def set(self, key: str, directives: list[ValidatedDirective]) -> None:
        if not settings.interpretation_cache_enabled:
            return
        serialized = serialize_directives(directives)
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (time.monotonic(), serialized)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._waiters: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        key: str,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        async with self._lock:
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(coro_factory())
                self._inflight[key] = task
                self._waiters[key] = 0
            self._waiters[key] += 1

        try:
            return await asyncio.shield(task)
        finally:
            async with self._lock:
                if key in self._waiters:
                    self._waiters[key] -= 1
                    if self._waiters[key] <= 0:
                        self._waiters.pop(key, None)
                        if self._inflight.get(key) is task:
                            self._inflight.pop(key, None)
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass
                    elif task.done():
                        if self._inflight.get(key) is task:
                            self._inflight.pop(key, None)

    def clear(self) -> None:
        self._inflight.clear()
        self._waiters.clear()


interpretation_cache = InterpretationCache()
single_flight = SingleFlight()
