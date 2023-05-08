from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, 'config')
DATA_DIR = Path(BASE_DIR, 'data')
NOTEBOOK_DIR = Path(BASE_DIR, 'notebooks')
MODEL_DIR = Path(BASE_DIR, 'models')
LOG_DIR = Path(BASE_DIR, 'logs')
