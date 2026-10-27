"""Lightweight, dependency-free MLIR text manipulation.

Only the small subset of MLIR needed by the Coral NPU build script is handled:
locating ``func.func`` definitions, replacing their bodies, concretising
dynamic dimensions, and reporting tensor sizes.  Keeping this in plain text
avoids a hard dependency on the IREE Python bindings, so the script runs with
nothing but the IREE command line tools installed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

try:  # pragma: no cover - exercised by both invocation styles
    from .tosa_mhsa import ArgSpec, parse_tensor_type
except ImportError:  # running as a plain script
    from tosa_mhsa import ArgSpec, parse_tensor_type

FUNC_KEYWORD = "func.func"
_ATTENTION_RE = re.compile(r"(attention|attn|mhsa|sdpa|flash)", re.IGNORECASE)

LINE_COMMENT = "//"


class MlirParseError(ValueError):
    """Raised when a function definition cannot be parsed."""


@dataclass
class FunctionDef:
    name: str
    start: int
    end: int  # one past the closing brace
    body_start: int  # offset of "{"
    body_end: int  # offset of the matching "}"
    visibility: str
    args: list[ArgSpec]
    result_type: str | None
    header_attrs: str
    source: str = field(default="", repr=False, compare=False)

    @property
    def signature(self) -> str:
        args = ", ".join(f"{a.name}: {a.type}" for a in self.args)
        text = f"func.func {self.visibility + ' ' if self.visibility else ''}@{self.name}({args})"
        if self.result_type is not None:
            text += f" -> {self.result_type}"
        if self.header_attrs:
            text += f" {self.header_attrs}"
        return text

    @property
    def text(self) -> str:
        return self.source[self.start : self.end]

    def looks_like_attention(self) -> bool:
        return bool(_ATTENTION_RE.search(self.name))


def _skip_ws(text: str, index: int) -> int:
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text.startswith(LINE_COMMENT, index):
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline + 1
            continue
        break
    return index


def _skip_string(text: str, index: int) -> int:
    quote = text[index]
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    raise MlirParseError("unterminated string literal")


def match_delimiter(text: str, index: int, opening: str, closing: str) -> int:
    """Return the offset of the delimiter matching ``text[index]``."""

    if index >= len(text) or text[index] != opening:
        raise MlirParseError(f"expected {opening!r} at offset {index}")
    depth = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _skip_string(text, index)
            continue
        if text.startswith(LINE_COMMENT, index):
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise MlirParseError(f"unbalanced {opening!r} starting at offset {index}")


def _parse_type(text: str, index: int) -> tuple[str, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise MlirParseError("unexpected end of input while parsing a type")
    char = text[index]
    if char in "([{<":
        closing = {"(": ")", "[": "]", "{": "}", "<": ">"}[char]
        end = match_delimiter(text, index, char, closing) + 1
        if char == "(":
            cursor = _skip_ws(text, end)
            if text.startswith("->", cursor):
                _, end = _parse_type(text, cursor + 2)
        return text[index:end], end
    match = re.match(r"[A-Za-z_!][\w.$]*", text[index:])
    if not match:
        raise MlirParseError(
            f"cannot parse type at offset {index}: {text[index:index + 24]!r}"
        )
    end = index + match.end()
    # ``tensor<...>`` / ``!tosa.shape<...>``: include the angle-bracketed body.
    cursor = _skip_ws(text, end)
    if cursor < len(text) and text[cursor] == "<":
        end = match_delimiter(text, cursor, "<", ">") + 1
    return text[index:end], end


def split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_arg(text: str) -> ArgSpec:
    name, separator, typ = text.partition(":")
    if not separator:
        raise MlirParseError(f"cannot parse function argument {text!r}")
    name = name.strip()
    if not name.startswith("%"):
        name = "%" + name
    return ArgSpec(name=name, type=typ.strip())


def _parse_function(text: str, start: int) -> FunctionDef:
    index = _skip_ws(text, start + len(FUNC_KEYWORD))
    visibility = ""
    for keyword in ("public", "private", "nested"):
        if re.match(rf"{keyword}\b", text[index:]):
            visibility = keyword
            index = _skip_ws(text, index + len(keyword))
            break
    if index >= len(text) or text[index] != "@":
        raise MlirParseError(f"expected a symbol name at offset {index}")
    match = re.match(r"[\w$.]+", text[index + 1 :])
    if not match:
        raise MlirParseError(f"empty symbol name at offset {index}")
    name = match.group(0)
    index = _skip_ws(text, index + 1 + len(name))

    if index >= len(text) or text[index] != "(":
        raise MlirParseError(f"expected argument list for @{name}")
    args_end = match_delimiter(text, index, "(", ")")
    args = [_parse_arg(part) for part in split_top_level(text[index + 1 : args_end])]
    index = _skip_ws(text, args_end + 1)

    result_type = None
    if text.startswith("->", index):
        result_type, index = _parse_type(text, index + 2)

    header_start = index
    while True:
        index = _skip_ws(text, index)
        if index >= len(text):
            raise MlirParseError(f"function @{name} has no body")
        if text[index] == "{":
            break
        if re.match(r"attributes\b", text[index:]):
            index = _skip_ws(text, index + len("attributes"))
            if index >= len(text) or text[index] != "{":
                raise MlirParseError(f"malformed attributes on @{name}")
            index = match_delimiter(text, index, "{", "}") + 1
            continue
        if text[index] in "([<":
            closing = {"(": ")", "[": "]", "<": ">"}[text[index]]
            index = match_delimiter(text, index, text[index], closing) + 1
            continue
        index += 1

    body_start = index
    body_end = match_delimiter(text, index, "{", "}")
    function = FunctionDef(
        name=name,
        start=start,
        end=body_end + 1,
        body_start=body_start,
        body_end=body_end,
        visibility=visibility,
        args=args,
        result_type=result_type,
        header_attrs=text[header_start:body_start].strip(),
        source=text,
    )
    return function


def find_functions(text: str) -> list[FunctionDef]:
    """Parse every ``func.func`` in ``text`` (bodies are not interpreted)."""

    functions: list[FunctionDef] = []
    for match in re.finditer(r"\bfunc\.func\b", text):
        try:
            functions.append(_parse_function(text, match.start()))
        except MlirParseError:
            continue
    return functions


def find_function(text: str, name: str) -> FunctionDef | None:
    for function in find_functions(text):
        if function.name == name:
            return function
    return None


def replace_function(text: str, function: FunctionDef, replacement: str) -> str:
    return text[: function.start] + replacement + text[function.end :]


def bind_dynamic_dims(text: str, value: int) -> tuple[str, int]:
    """Replace ``?`` extents inside tensor types with a concrete value."""

    if value < 1:
        raise ValueError("bound value must be >= 1")
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        body = match.group(0)
        count += body.count("?")
        return body.replace("?", str(value))

    bound = re.sub(r"tensor<[^<>]*>", _replace, text)
    return bound, count


def dynamic_extent_count(text: str) -> int:
    return sum(
        match.group(0).count("?") for match in re.finditer(r"tensor<[^<>]*>", text)
    )


@dataclass
class TensorStat:
    type: str
    count: int
    bytes_per_instance: int

    @property
    def total_bytes(self) -> int:
        return self.count * self.bytes_per_instance


def tensor_stats(text: str) -> list[TensorStat]:
    """Largest fully static tensor types, by instance size."""

    counter = Counter(re.findall(r"tensor<[0-9x]+(?:bf16|f16|f32|f64)>", text))
    stats: list[TensorStat] = []
    for type_name, count in counter.items():
        try:
            shape, dtype = parse_tensor_type(type_name)
        except ValueError:
            continue
        size = 1
        for dim in shape:
            size *= dim
        stats.append(
            TensorStat(
                type_name,
                count,
                size * {"f64": 8, "f32": 4, "f16": 2, "bf16": 2}[dtype],
            )
        )
    stats.sort(key=lambda item: item.bytes_per_instance, reverse=True)
    return stats


def attention_candidates(functions: list[FunctionDef]) -> list[FunctionDef]:
    candidates = []
    for function in functions:
        if len(function.args) not in (3, 4):
            continue
        try:
            ranks = {len(arg.shape) for arg in function.args[:3]}
        except ValueError:
            continue
        if ranks in ({3}, {4}) and function.looks_like_attention():
            candidates.append(function)
    return candidates
