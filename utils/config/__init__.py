from utils.config.yml_handler import read_yaml

try:
    db_cfg = read_yaml("utils/db_config.yaml") or {}
except FileNotFoundError:
    db_cfg = {}
