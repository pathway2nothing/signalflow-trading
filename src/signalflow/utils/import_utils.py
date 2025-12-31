def import_model_class(class_path: str) -> type:
    """Dynamically import model class from string path."""
    parts = class_path.rsplit(".", 1)
    if len(parts) == 2:
        module_name, class_name = parts
    else:
        raise ValueError(f"Invalid class path: {class_path}")

    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
