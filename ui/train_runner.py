import os
import sys
import time
import json
import subprocess
from typing import List, Optional, Dict, Any

try:
    import psutil
except ImportError:
    psutil = None

LOCK_FILE = os.path.join("results", "train.lock")
LOG_FILE = os.path.join("results", "train_log.txt")


def _is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is running."""
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    else:
        # OS fallback check
        try:
            os.kill(pid, 0)
            return True
        except (OSError, AttributeError):
            return False


def is_running() -> bool:
    """
    Checks if a training run is currently active.
    Returns True if train.lock exists and its PID is alive.
    Removes stale lock file and returns False if PID is dead.
    """
    if not os.path.exists(LOCK_FILE):
        return False

    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = data.get("pid")
        if pid and _is_pid_alive(pid):
            return True
    except Exception:
        pass

    # Stale lock cleanup
    clean_lock()
    return False


def get_lock_info() -> Optional[Dict[str, Any]]:
    """Reads lock file data if present."""
    if not os.path.exists(LOCK_FILE):
        return None
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clean_lock():
    """Removes train.lock file if present."""
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass


def start_run(script_name: str, args: List[str] = None) -> subprocess.Popen:
    """
    Launches python script as a background process with unbuffered output (-u).
    Redirects stdout and stderr to results/train_log.txt.
    Writes process info to results/train.lock.
    """
    if is_running():
        raise RuntimeError("A training run is already in progress.")

    if args is None:
        args = []

    os.makedirs("results", exist_ok=True)

    cmd = [sys.executable, "-u", script_name] + [str(a) for a in args]

    # Open train_log.txt for writing (overwriting previous run log)
    log_fd = open(LOG_FILE, "w", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    lock_data = {
        "pid": proc.pid,
        "start_time": time.time(),
        "script_name": script_name,
        "args": args,
        "cmd": cmd,
    }

    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)

    return proc


def stop_run() -> bool:
    """
    Terminates the currently running training process (if active) and cleans up lock file.
    Returns True if a process was stopped.
    """
    if not os.path.exists(LOCK_FILE):
        return False

    pid = None
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            pid = data.get("pid")
    except Exception:
        pass

    stopped = False
    if pid and _is_pid_alive(pid):
        if psutil is not None:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                stopped = True
            except Exception:
                pass
        else:
            try:
                os.kill(pid, 9)
                stopped = True
            except Exception:
                pass

    clean_lock()
    return stopped


def tail_log(n_lines: int = 40) -> str:
    """Returns the last N lines of results/train_log.txt safely."""
    if not os.path.exists(LOG_FILE):
        return "Log file not created yet. Waiting for output..."

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            if not lines:
                return "Training log started... waiting for output."
            return "".join(lines[-n_lines:])
    except Exception as e:
        return f"Error reading log file: {e}"
