"""Tool registry.

Tools are plain Python functions. The JSON Schema handed to the model is
derived from the signature and the docstring, so adding a capability is just::

    @tool(category="media")
    def play_song(path: str, volume: int = 50) -> str:
        '''Play an audio file.

        Args:
            path: File to play.
            volume: 0-100.
        '''
        ...

The agent, the CLI and every provider share this one registry.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin

log = logging.getLogger("jarvis.tools")

_REGISTRY: Dict[str, "Tool"] = {}


class ToolError(Exception):
    """Raised by a tool to signal a clean, reportable failure."""


@dataclass
class ToolResult:
    """Outcome of running a tool."""

    output: str = ""
    ok: bool = True
    data: Any = None

    def to_text(self) -> str:
        if self.output:
            return self.output
        if self.data is None:
            return "ok" if self.ok else "failed"
        if isinstance(self.data, str):
            return self.data
        try:
            return json.dumps(self.data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(self.data)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    confirm: bool = False
    category: str = "general"
    #: Names this tool is exposed under to the model.
    aliases: List[str] = field(default_factory=list)

    def spec(self):
        from ..providers.base import ToolSpec

        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def __call__(self, **kwargs: Any) -> ToolResult:
        return run_tool(self.name, kwargs)


# -- schema generation ------------------------------------------------------
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _type_schema(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is typing.Union or origin is getattr(__import__("types"), "UnionType", object):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_schema(non_none[0])
        return {"anyOf": [_type_schema(a) for a in non_none]} if non_none else {"type": "string"}
    if origin in (list, typing.List) or origin is typing.Sequence:
        item = _type_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if origin is typing.Literal and args:
        return {"type": "string", "enum": [str(a) for a in args]}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    if isinstance(annotation, type) and issubclass(annotation, bool):
        return {"type": "boolean"}
    if isinstance(annotation, type) and issubclass(annotation, int):
        return {"type": "integer"}
    if isinstance(annotation, type) and issubclass(annotation, float):
        return {"type": "number"}
    return {"type": "string"}


def _parse_docstring(doc: Optional[str]) -> Dict[str, Any]:
    """Extract a summary and per-parameter descriptions from a docstring."""
    if not doc:
        return {"summary": "", "params": {}}
    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: List[str] = []
    params: Dict[str, str] = {}
    section = None
    current: Optional[str] = None
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        header = stripped.lower().rstrip(":")
        if header in {"args", "arguments", "parameters"}:
            section = "args"
            current = None
            continue
        if header in {"returns", "return", "raises", "example", "examples", "note", "notes"}:
            section = None
            continue
        if section == "args":
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
            if match:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current and stripped:
                params[current] = (params[current] + " " + stripped).strip()
            continue
        if stripped and section is None and not summary_lines:
            summary_lines.append(stripped)
    summary = " ".join(summary_lines).strip()
    return {"summary": summary, "params": params}


def build_parameters(func: Callable[..., Any]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # pragma: no cover - unresolvable forward refs
        hints = {}
    doc = _parse_docstring(func.__doc__)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, parameter.annotation)
        schema = _type_schema(annotation)
        description = doc["params"].get(name)
        if description:
            schema["description"] = description
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = parameter.default if isinstance(parameter.default, (str, int, float, bool)) else None
            if schema.get("default") is None:
                schema.pop("default", None)
        properties[name] = schema
    parameters: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return parameters


# -- decorator / registry ----------------------------------------------------
def tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    confirm: bool = False,
    category: str = "general",
    aliases: Optional[List[str]] = None,
):
    """Register *func* as a tool the model can call."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        doc = _parse_docstring(func.__doc__)
        tool_name = name or func.__name__
        tool_desc = description or doc["summary"] or "Tool %s" % tool_name
        entry = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=build_parameters(func),
            func=func,
            confirm=confirm,
            category=category,
            aliases=list(aliases or []),
        )
        _REGISTRY[tool_name] = entry
        func.tool = entry  # type: ignore[attr-defined]
        return func

    return decorate(_func) if _func is not None else decorate


def register_tool(entry: Tool) -> Tool:
    _REGISTRY[entry.name] = entry
    return entry


def get(name: str) -> Optional[Tool]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    for entry in _REGISTRY.values():
        if name in entry.aliases:
            return entry
    return None


def all_tools() -> List[Tool]:
    return sorted(_REGISTRY.values(), key=lambda t: (t.category, t.name))


def specs(names: Optional[List[str]] = None, disabled: Optional[List[str]] = None):
    """Return :class:`ToolSpec` objects, optionally filtered by name."""
    disabled = set(disabled or ())
    selected = []
    for entry in all_tools():
        if names and entry.name not in names:
            continue
        if entry.name in disabled:
            continue
        selected.append(entry.spec())
    return selected


def run_tool(name: str, arguments: Dict[str, Any]) -> ToolResult:
    """Execute a tool by name with a dict of arguments."""
    entry = get(name)
    if entry is None:
        return ToolResult(output="unknown tool %r" % name, ok=False)
    args = dict(arguments or {})
    if "__raw__" in args and len(args) == 1:
        return ToolResult(
            output="arguments were not valid JSON: %s" % args["__raw__"], ok=False
        )
    signature = inspect.signature(entry.func)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if not accepts_kwargs:
        allowed = set(signature.parameters)
        unknown = set(args) - allowed
        if unknown:
            log.debug("dropping unknown arguments for %s: %s", name, unknown)
            args = {k: v for k, v in args.items() if k in allowed}
    try:
        result = entry.func(**args)
    except ToolError as exc:
        return ToolResult(output=str(exc), ok=False)
    except TypeError as exc:
        return ToolResult(
            output="bad arguments for %s: %s (got %s)" % (name, exc, sorted(args)), ok=False
        )
    except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
        log.debug("tool %s raised", name, exc_info=True)
        return ToolResult(output="%s: %s" % (type(exc).__name__, exc), ok=False)
    if isinstance(result, ToolResult):
        return result
    if result is None:
        return ToolResult(output="ok", ok=True)
    if isinstance(result, (dict, list, tuple, int, float, bool)):
        return ToolResult(data=result, output=_dump(result))
    return ToolResult(output=str(result), ok=True)


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
