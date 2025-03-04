import torch
import chess
import chess.pgn
import chess.engine
import datetime
import io
import numpy as np
import random
import os
import time
import subprocess
import platform
import shutil
import glob
from tqdm import tqdm

# Import UCI_MOVES mapping from the main file
from utils.paste import UCI_MOVES
