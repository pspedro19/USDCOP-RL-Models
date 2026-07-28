"""Static FABRIC contract validator used by CI and local readiness checks."""
from pathlib import Path

legacy_bypass_allowlist = Path("config/metrics/legacy_bypass_allowlist.yaml")


def validate() -> list[str]:
    return [] if legacy_bypass_allowlist.exists() else [str(legacy_bypass_allowlist)]


if __name__ == "__main__":
    missing = validate()
    if missing:
        raise SystemExit("missing required FABRIC contracts: " + ", ".join(missing))
