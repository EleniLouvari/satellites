"""Interrupt one isolated analysis process tree without stopping its supervisor."""

import os
import signal
import time
from contextlib import contextmanager

import psutil


@contextmanager
def task_group():
    """Retain Windows descendant ownership across task-leader exit."""
    group = None
    if os.name == "nt":
        from .windows_job import WindowsJob

        group = WindowsJob()
    try:
        yield group
    finally:
        if group is not None:
            group.close()


def isolate_task(group_name=None):
    """Isolate all task descendants before any analysis work starts."""
    if os.name == "posix":
        os.setsid()
    else:
        from .windows_job import join_job

        if group_name is None:
            raise RuntimeError("Error: Windows analysis requires a supervisor-owned job.")
        join_job(group_name)


def _stop_posix_group(pid):
    """Freeze the task group to prevent new descendants, then kill only that group."""
    try:
        os.killpg(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return []
    members = []
    try:
        for process in psutil.process_iter():
            try:
                if os.getpgid(process.pid) == pid:
                    members.append(process)
            except ProcessLookupError:
                continue
    finally:
        # This PID is the dedicated session leader established by isolate_task.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return members


def _can_write(process):
    """Check process identity and exclude exited processes, including zombies."""
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _active_members(members, group):
    """Check kernel job accounting or identity-aware POSIX process handles."""
    if group is not None:
        return bool(group.active())
    return any(_can_write(process) for process in members)


def stop_task(child, group=None):
    """Wait for the selected task and its descendants to stop before reconciliation."""
    if os.name == "posix":
        members = _stop_posix_group(child.pid)
    else:
        if group is None:
            raise RuntimeError("Error: Cannot stop Windows analysis without its owned job.")
        group.terminate()
        members = []
    child.join(timeout=10)
    deadline = time.monotonic() + 10
    # Polling also works on Docker kernels without psutil's pidfd wait support.
    while _active_members(members, group) and time.monotonic() < deadline:
        time.sleep(0.05)
    if child.is_alive() or _active_members(members, group):
        raise RuntimeError("Error: Analysis processes have not exited; refusing to release the run.")
