# docs/hooks.py
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


def on_pre_page_macros(env):
    """Skip macro processing for notebook pages to avoid Jinja syntax conflicts."""
    page = env.page
    if page and page.file.src_path.endswith(".ipynb"):
        return False  # Skip macros for notebooks
    return True


def on_post_page_macros(env):
    """Post-processing hook (no-op)."""
    pass


def on_post_build(env):
    """Post-build hook (no-op)."""
    pass
