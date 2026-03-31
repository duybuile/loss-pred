try:
    import tomllib as _toml_loader
except ModuleNotFoundError:  # pragma: no cover - exercised only on older Python versions
    import toml as _toml_loader


# Load the TOML configuration file
def load_toml(file_path):
    if _toml_loader.__name__ == "tomllib":
        with open(file_path, "rb") as f:
            config = _toml_loader.load(f)
    else:
        with open(file_path, "r") as f:
            config = _toml_loader.load(f)
    return config
