"""让 Module_A 能找到项目原有的公共代码，不复制 CIGPose。"""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent.parent
PUBLIC = PROJECT / 'mmpose/analysis/fyp'
if str(PUBLIC) not in sys.path:
    sys.path.insert(0, str(PUBLIC))
