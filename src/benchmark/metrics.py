import time
import os
import psutil
import torch
import jiwer
from typing import Dict, Any, Tuple

def get_memory_usage() -> Tuple[float, float]:
    """
    Measure current CPU RSS memory (in MB) and GPU memory allocated (in MB).
    """
    # CPU Memory (Resident Set Size in MB)
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # GPU Memory (in MB)
    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)

    return round(cpu_mem_mb, 2), round(gpu_mem_mb, 2)

def calculate_wer(reference: str, prediction: str) -> float:
    """
    Calculate Word Error Rate (WER) using jiwer.
    Handles empty strings gracefully.
    """
    ref = str(reference).strip()
    pred = str(prediction).strip()
    
    if not ref:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
        
    try:
        score = jiwer.wer(ref, pred)
        return round(float(score), 4)
    except Exception:
        return 1.0

def compute_rtf(inference_time_sec: float, audio_duration_sec: float) -> float:
    """
    Compute Real-Time Factor: RTF = inference_time / audio_duration.
    """
    if audio_duration_sec <= 0:
        return 0.0
    return round(float(inference_time_sec / audio_duration_sec), 4)
