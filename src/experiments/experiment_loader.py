"""
Experiment Loader
=================

Load and validate experiment configurations from YAML files.
Supports JSON Schema validation and Pydantic model validation.

Usage:
    from src.experiments import load_experiment_config, validate_experiment_config

    # Load and validate in one step
    config = load_experiment_config("config/experiments/my_experiment.yaml")

    # Validate only (returns list of errors)
    errors = validate_experiment_config("config/experiments/my_experiment.yaml")
    if errors:
        print(f"Validation failed: {errors}")

Author: Trading Team
Date: 2026-01-17
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import yaml

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from pydantic import ValidationError

from .experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_EXPERIMENTS_DIR = Path("config/experiments")
DEFAULT_SCHEMA_PATH = Path("config/schemas/experiment.schema.json")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(schema_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load JSON Schema for validation.

    Args:
        schema_path: Path to schema file

    Returns:
        Schema dict or None if not available
    """
    if schema_path is None:
        schema_path = DEFAULT_SCHEMA_PATH

    if not schema_path.exists():
        logger.warning(f"Schema not found: {schema_path}")
        return None

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_with_jsonschema(
    config_dict: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Validate configuration against JSON Schema.

    Args:
        config_dict: Configuration dictionary
        schema: JSON Schema (loads default if None)

    Returns:
        List of validation error messages
    """
    if not JSONSCHEMA_AVAILABLE:
        logger.warning("jsonschema not installed, skipping schema validation")
        return []

    if schema is None:
        schema = load_schema()

    if schema is None:
        return []

    errors = []
    validator = jsonschema.Draft202012Validator(schema)

    for error in validator.iter_errors(config_dict):
        path = " -> ".join(str(p) for p in error.absolute_path)
        errors.append(f"{path}: {error.message}" if path else error.message)

    return errors


def validate_with_pydantic(config_dict: Dict[str, Any]) -> List[str]:
    """
    Validate configuration with Pydantic models.

    Args:
        config_dict: Configuration dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    try:
        ExperimentConfig(**config_dict)
    except ValidationError as e:
        for error in e.errors():
            path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{path}: {error['msg']}")

    return errors


def validate_experiment_config(
    config_path: Union[str, Path],
    strict: bool = True,
) -> List[str]:
    """
    Validate experiment configuration file.

    Performs both JSON Schema and Pydantic validation.

    Args:
        config_path: Path to YAML config file
        strict: If True, validate with both schema and Pydantic

    Returns:
        List of validation error messages (empty if valid)

    Example:
        errors = validate_experiment_config("config/experiments/my_exp.yaml")
        if not errors:
            print("Configuration is valid!")
        else:
            for error in errors:
                print(f"Error: {error}")
    """
    all_errors = []

    # Load YAML
    try:
        config_dict = load_yaml(config_path)
    except FileNotFoundError as e:
        return [str(e)]
    except yaml.YAMLError as e:
        return [f"YAML parsing error: {e}"]

    # JSON Schema validation
    if strict:
        schema_errors = validate_with_jsonschema(config_dict)
        all_errors.extend(schema_errors)

    # Pydantic validation
    pydantic_errors = validate_with_pydantic(config_dict)
    all_errors.extend(pydantic_errors)

    return all_errors


def load_experiment_config(
    config_path: Union[str, Path],
    validate: bool = True,
) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        validate: Whether to validate before loading

    Returns:
        ExperimentConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If validation fails
        yaml.YAMLError: If YAML is invalid

    Example:
        config = load_experiment_config("config/experiments/baseline_ppo_v1.yaml")
        print(f"Experiment: {config.experiment.name}")
        print(f"Total timesteps: {config.training.total_timesteps}")
    """
    config_path = Path(config_path)

    # Load YAML
    config_dict = load_yaml(config_path)

    # Validate if requested
    if validate:
        errors = validate_experiment_config(config_path, strict=True)
        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")

    # Parse with Pydantic
    try:
        config = ExperimentConfig(**config_dict)
        logger.info(f"Loaded experiment config: {config.experiment.name}")
        return config
    except ValidationError as e:
        raise ValueError(f"Failed to parse configuration: {e}")


def list_available_experiments(
    experiments_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    List all available experiment configurations.

    Args:
        experiments_dir: Directory containing experiment YAML files

    Returns:
        List of experiment info dicts with name, version, path, valid status

    Example:
        experiments = list_available_experiments()
        for exp in experiments:
            status = "valid" if exp["valid"] else "invalid"
            print(f"{exp['name']} v{exp['version']} - {status}")
    """
    if experiments_dir is None:
        experiments_dir = DEFAULT_EXPERIMENTS_DIR

    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        logger.warning(f"Experiments directory not found: {experiments_dir}")
        return []

    experiments = []

    for yaml_file in experiments_dir.glob("*.yaml"):
        try:
            config_dict = load_yaml(yaml_file)
            exp_info = config_dict.get("experiment", {})

            # Quick validation
            errors = validate_with_pydantic(config_dict)

            experiments.append({
                "name": exp_info.get("name", yaml_file.stem),
                "version": exp_info.get("version", "unknown"),
                "description": exp_info.get("description", ""),
                "tags": exp_info.get("tags", []),
                "path": str(yaml_file),
                "valid": len(errors) == 0,
                "errors": errors if errors else None,
            })

        except Exception as e:
            experiments.append({
                "name": yaml_file.stem,
                "version": "unknown",
                "path": str(yaml_file),
                "valid": False,
                "errors": [str(e)],
            })

    # Sort by name
    experiments.sort(key=lambda x: x["name"])

    return experiments


def get_experiment_by_name(
    name: str,
    experiments_dir: Optional[Path] = None,
) -> Optional[ExperimentConfig]:
    """
    Get experiment configuration by name.

    Searches for experiment with matching name in experiments directory.

    Args:
        name: Experiment name to search for
        experiments_dir: Directory to search in

    Returns:
        ExperimentConfig if found, None otherwise
    """
    experiments = list_available_experiments(experiments_dir)

    for exp in experiments:
        if exp["name"] == name and exp["valid"]:
            return load_experiment_config(exp["path"])

    return None


def create_experiment_from_template(
    name: str,
    template: str = "baseline_ppo_v1",
    output_path: Optional[Path] = None,
    **overrides,
) -> Path:
    """
    Create new experiment config from template.

    Args:
        name: New experiment name
        template: Template experiment name
        output_path: Output file path
        **overrides: Config values to override

    Returns:
        Path to created config file
    """
    # Load template
    template_config = get_experiment_by_name(template)
    if template_config is None:
        raise ValueError(f"Template experiment not found: {template}")

    # Convert to dict and modify
    config_dict = template_config.to_dict()
    config_dict["experiment"]["name"] = name

    # Apply overrides
    for key, value in overrides.items():
        parts = key.split(".")
        target = config_dict
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    # Determine output path
    if output_path is None:
        output_path = DEFAULT_EXPERIMENTS_DIR / f"{name}.yaml"

    # Write new config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created experiment config: {output_path}")
    return output_path


__all__ = [
    "load_experiment_config",
    "validate_experiment_config",
    "list_available_experiments",
    "get_experiment_by_name",
    "create_experiment_from_template",
]
