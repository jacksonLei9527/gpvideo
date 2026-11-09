"""Cooperative job cancellation with termination of active FFmpeg processes."""
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


class JobCancelled(Exception):
    """Raised at pipeline checkpoints after a user requests cancellation."""


_LOCK = threading.RLock()
_EVENTS: dict[str, threading.Event] = {}
_PROCESSES: dict[str, set[subprocess.Popen]] = {}


def register_job(job_id: str) -> None:
    with _LOCK:
        _EVENTS[job_id] = threading.Event()
        _PROCESSES[job_id] = set()


def unregister_job(job_id: str) -> None:
    with _LOCK:
        _EVENTS.pop(job_id, None)
        _PROCESSES.pop(job_id, None)


def _job_id(workdir) -> str | None:
    if workdir is None:
        return None
    return Path(workdir).name


def is_cancelled(workdir) -> bool:
    job_id = _job_id(workdir)
    if not job_id:
        return False
    with _LOCK:
        event = _EVENTS.get(job_id)
    return bool(event and event.is_set())


def check_cancel(workdir) -> None:
    if is_cancelled(workdir):
        raise JobCancelled("Cancelled by user")


def _signal_process(proc: subprocess.Popen, sig: int) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, sig)
        else:
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _signal_process(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _signal_process(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def request_cancel(job_id: str) -> bool:
    with _LOCK:
        event = _EVENTS.get(job_id)
        processes = list(_PROCESSES.get(job_id, ()))
        if event is None:
            return False
        event.set()
    for proc in processes:
        _terminate_process(proc)
    return True


def run_command(cmd, *, workdir=None, cwd=None, timeout=None,
                capture_output=True, text=True) -> subprocess.CompletedProcess:
    """Run a command while polling its job's cancel event and capturing output."""
    check_cancel(workdir)
    job_id = _job_id(workdir)
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=stdout, stderr=stderr, text=text,
        start_new_session=(os.name == "posix"),
    )
    if job_id:
        with _LOCK:
            _PROCESSES.setdefault(job_id, set()).add(proc)

    deadline = time.monotonic() + timeout if timeout is not None else None
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if is_cancelled(workdir):
                    _terminate_process(proc)
                    proc.communicate()
                    raise JobCancelled("Cancelled by user")
                if deadline is not None and time.monotonic() >= deadline:
                    _terminate_process(proc)
                    raise subprocess.TimeoutExpired(cmd, timeout)
        check_cancel(workdir)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    finally:
        if job_id:
            with _LOCK:
                _PROCESSES.get(job_id, set()).discard(proc)
