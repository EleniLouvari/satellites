"""Retain Windows task descendants even after their original parent exits."""

import ctypes
from ctypes import wintypes
from uuid import uuid4


class _BasicLimits(ctypes.Structure):
    """Windows JOBOBJECT_BASIC_LIMIT_INFORMATION layout."""

    _fields_ = [
        ("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    """Windows JOBOBJECT_EXTENDED_LIMIT_INFORMATION layout."""

    _fields_ = [
        ("basic", _BasicLimits), ("io", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    """Windows JOBOBJECT_BASIC_ACCOUNTING_INFORMATION layout."""

    _fields_ = [
        ("times", ctypes.c_int64 * 4), ("faults", wintypes.DWORD),
        ("total", wintypes.DWORD), ("active", wintypes.DWORD), ("terminated", wintypes.DWORD),
    ]


def _kernel():
    """Declare pointer-safe Win32 signatures without adding a dependency."""
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "OpenJobObjectW": ([wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
        "GetCurrentProcess": ([], wintypes.HANDLE),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        "QueryInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL,
        ),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
    return api


def _checked(result):
    """Fail closed when Windows cannot establish or terminate task ownership."""
    if not result:
        raise OSError(f"Error: Windows analysis job operation failed: {ctypes.WinError(ctypes.get_last_error())}")
    return result


def join_job(name):
    """Attach the task before it can spawn children; only the supervisor retains a handle."""
    api = _kernel()
    handle = _checked(api.OpenJobObjectW(0x0001, False, name))  # JOB_OBJECT_ASSIGN_PROCESS
    try:
        _checked(api.AssignProcessToJobObject(handle, api.GetCurrentProcess()))
    finally:
        api.CloseHandle(handle)


class WindowsJob:
    """Own a task's kernel-managed descendant set independently of its leader PID."""

    def __init__(self):
        """Create a private job that kills descendants if the supervisor disappears."""
        self.api = _kernel()
        self.name = f"Local\\parcel-analysis-{uuid4()}"
        self.handle = _checked(self.api.CreateJobObjectW(None, self.name))
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            _checked(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def terminate(self):
        """Terminate every associated process, including orphaned descendants."""
        _checked(self.api.TerminateJobObject(self.handle, 1))

    def active(self):
        """Count processes that have not finished exiting from this job."""
        info = _Accounting()
        _checked(self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.active

    def close(self):
        """Release the supervisor handle, also containing failures during startup."""
        if self.handle:
            _checked(self.api.CloseHandle(self.handle))
            self.handle = None
