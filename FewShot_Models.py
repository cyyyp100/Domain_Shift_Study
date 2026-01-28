import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Chemins
EMBEDDINGS_ROOT = "/Users/bilaldelais/Desktop/Etude Technique /Code/embeddings"
OUTPUT_ROOT = "/Users/bilaldelais/Desktop/Etude Technique /Code/Résultats"


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

#chargement des données 
XPaths = list(Path(EMBEDDINGS_ROOT).glob("*.npy"))

X = np.load(XPaths[0])
print(f"Loaded embeddings shape: {X.shape}")
