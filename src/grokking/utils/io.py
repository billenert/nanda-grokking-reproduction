"""Configuration loading and I/O utilities.

Implements config loading with CLI override support using --set key=value dot-path notation.
"""

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        path: Path to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _set_nested_value(d: dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot-path notation.

    Args:
        d: Dictionary to modify (in-place).
        key_path: Dot-separated path to the key (e.g., "model.d_model").
        value: Value to set.

    Raises:
        KeyError: If any intermediate key doesn't exist.
    """
    keys = key_path.split(".")
    current = d

    for key in keys[:-1]:
        if key not in current:
            raise KeyError(f"Key '{key}' not found in config path '{key_path}'")
        current = current[key]

    final_key = keys[-1]
    if final_key not in current:
        raise KeyError(f"Key '{final_key}' not found in config path '{key_path}'")

    current[final_key] = value


def _parse_value(value_str: str) -> Any:
    """Parse a string value into the appropriate Python type.

    Args:
        value_str: String value to parse.

    Returns:
        Parsed value (int, float, bool, None, list, or string).
    """
    # Handle None
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None

    # Handle booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Handle lists (JSON format)
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass

    # Handle integers
    try:
        return int(value_str)
    except ValueError:
        pass

    # Handle floats (including scientific notation)
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides to a configuration dictionary.

    Args:
        config: Base configuration dictionary.
        overrides: List of override strings in "key=value" format.

    Returns:
        New configuration dictionary with overrides applied.

    Raises:
        ValueError: If override format is invalid.
        KeyError: If override key path doesn't exist in config.
    """
    config = copy.deepcopy(config)

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: '{override}'. Expected 'key=value'."
            )

        key_path, value_str = override.split("=", 1)
        value = _parse_value(value_str)
        _set_nested_value(config, key_path, value)

    return config


def save_jsonl(data: list[dict], path: str | Path) -> None:
    """Append data to a JSONL file (one JSON object per line).

    Args:
        data: List of dictionaries to write.
        path: Path to the JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load data from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries loaded from the file.
    """
    path = Path(path)
    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data
