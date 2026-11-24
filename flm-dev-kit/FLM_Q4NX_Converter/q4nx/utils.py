# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import os

def get_relativeL2(y: np.ndarray, y_ref: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y - y_ref) ** 2))
    ref_norm = np.sqrt(np.mean(y_ref ** 2))
    return rmse / ref_norm

def get_relativeL1(y: np.ndarray, y_ref: np.ndarray) -> float:
    l1 = np.sum(np.abs(y - y_ref))
    ref_sum = np.sum(np.abs(y_ref))
    return l1 / ref_sum

def get_rmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    return np.sqrt(np.mean((y - y_ref) ** 2)    ) 

def get_cosine_similarity(y: np.ndarray, y_ref: np.ndarray) -> float:
    # squize to 1D array
    y = y.reshape(-1)
    y_ref = y_ref.reshape(-1)
    dot_product = np.dot(y, y_ref)
    norm_y = np.linalg.norm(y)
    norm_y_ref = np.linalg.norm(y_ref)
    
    return dot_product / (norm_y * norm_y_ref)


def round_up_to_multiple(n: int, multiple: int) -> int:
    if multiple == 0:
        return n  # avoid division by zero
    return ((n + multiple - 1) // multiple) * multiple

def create_dir_if_not_exists(path: str):
    if not os.path.exists(path):
        print(f"[INFO] Creating directory {path}...")
        os.makedirs(path)