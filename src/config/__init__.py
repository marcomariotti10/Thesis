import sys
import os
from pathlib import Path

# Add scripts directory to path to import settings.py
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from settings import *
from .directories import *
from .helpers import *
from .model_architectures import *