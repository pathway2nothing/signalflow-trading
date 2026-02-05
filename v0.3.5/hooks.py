# docs/hooks.py
import os
import tomllib


def define_env(env):
    """
    Це хук для mkdocs-macros-plugin
    """
    toml_path = "pyproject.toml"

    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
            version = data["project"]["version"]

            env.variables.project_version = version

            env.variables.project_name = data["project"]["name"]
    except FileNotFoundError:
        env.variables.project_version = "unknown"
