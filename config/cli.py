"""Load per-entry-point JSON defaults and persist effective launch arguments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"


def default_config_path(script_file: str | Path | None = None) -> Path:
    """Return ``config/<entrypoint-relative-path>.json`` for a CLI script."""
    path = Path(script_file or sys.argv[0]).resolve()
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = Path(path.name)
    return (CONFIG_ROOT / relative).with_suffix(".json")


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    script_file: str | Path | None = None,
) -> argparse.Namespace:
    """Apply JSON defaults before parsing CLI overrides.

    Configs may be ``{"defaults": {...}}`` or a plain mapping. Explicit CLI
    values always take precedence. Required options can be supplied by config.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    default_path = default_config_path(script_file)
    explicit_path = _find_config_argument(args)
    selected_path = Path(explicit_path).expanduser() if explicit_path else default_path

    if not any(action.dest == "config_file" for action in parser._actions):
        parser.add_argument(
            "--config",
            dest="config_file",
            default=str(selected_path),
            help="JSON config file (defaults to the matching file under config/).",
        )
    else:
        parser.set_defaults(config_file=str(selected_path))

    if selected_path.exists():
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config must contain a JSON object: {selected_path}")
        defaults = data.get("defaults", data)
        if not isinstance(defaults, dict):
            raise ValueError(f"Config 'defaults' must be a JSON object: {selected_path}")
        _apply_defaults(parser, defaults, selected_path)
    else:
        source = "explicit" if explicit_path else "default"
        raise FileNotFoundError(f"{source.capitalize()} config file not found: {selected_path}")

    return parser.parse_args(args)


def save_startup_config(args: argparse.Namespace | dict[str, Any], checkpoint_dir: str | Path) -> Path:
    """Write effective startup parameters into the checkpoint directory."""
    values = vars(args).copy() if isinstance(args, argparse.Namespace) else dict(args)
    config_source = values.get("config_file")
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(config_source) if config_source is not None else None,
        "command_line": sys.argv,
        "arguments": _jsonable(values),
    }
    target = Path(checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)
    output_path = target / "startup_config.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _find_config_argument(args: Sequence[str]) -> str | None:
    for index, value in enumerate(args):
        if value == "--config":
            if index + 1 >= len(args):
                raise ValueError("--config requires a path")
            return args[index + 1]
        if value.startswith("--config="):
            return value.split("=", 1)[1]
    return None


def _apply_defaults(
    parser: argparse.ArgumentParser,
    defaults: dict[str, Any],
    config_path: Path,
) -> None:
    actions_by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.dest != argparse.SUPPRESS:
            actions_by_dest[action.dest] = action
    unknown = sorted(set(defaults) - set(actions_by_dest))
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown)}")

    converted: dict[str, Any] = {}
    for key, value in defaults.items():
        # JSON null keeps the parser's own default. This is useful for
        # runtime-dependent defaults such as selecting CUDA when available.
        if value is None:
            continue
        action = actions_by_dest[key]
        converted[key] = _convert_config_value(action, value)
        if action.choices is not None:
            values_to_check = converted[key] if isinstance(converted[key], list) else [converted[key]]
            invalid = [item for item in values_to_check if item is not None and item not in action.choices]
            if invalid:
                raise ValueError(
                    f"Invalid value for '{key}' in {config_path}: {invalid}; "
                    f"expected one of {list(action.choices)}"
                )
        if action.required and value is not None:
            action.required = False
    parser.set_defaults(**converted)


def _convert_config_value(action: argparse.Action, value: Any) -> Any:
    if value is None or action.type is None:
        return value
    if isinstance(value, list) and action.nargs not in (None, "?", 1):
        return [action.type(item) for item in value]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return action.type(value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

