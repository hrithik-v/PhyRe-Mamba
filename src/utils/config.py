import os
import yaml


def load_config(path):
	if not os.path.exists(path):
		raise FileNotFoundError(f"Config not found: {path}")
	with open(path, 'r', encoding='utf-8') as f:
		data = yaml.safe_load(f) or {}
	return data
