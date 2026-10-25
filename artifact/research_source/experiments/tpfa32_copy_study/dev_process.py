"""Small Windows process identity/job helpers; no service or privilege changes."""
import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path
from core import write


def publish_handshake(path,value):
    """Publish a new immutable handshake only after its bytes are durable."""
    path=Path(path);pending=path.with_suffix('.pending')
    write(pending,value)
    os.rename(pending,path)


def kernel():return ctypes.WinDLL('kernel32',use_last_error=True)


def process_identity(pid):
    k=kernel();k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
    k.CloseHandle.argtypes=[w.HANDLE]
    handle=k.OpenProcess(0x1000|0x00100000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return None
        raise OSError(error,'process identity/ownership unresolved')
    try:
        k.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.POINTER(w.FILETIME)]*4
        creation,exit_time,system,user=(w.FILETIME() for _ in range(4))
        if not k.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(system),ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        k.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD]
        status=k.WaitForSingleObject(handle,0)
        if status==0:return None
        if status!=258:raise RuntimeError('ambiguous process wait status')
        return dict(pid=pid,creation_filetime=(creation.dwHighDateTime<<32)|creation.dwLowDateTime)
    finally:k.CloseHandle(handle)


def require_dead(record):
    if record is None:raise RuntimeError('prior process identity missing')
    current=process_identity(record['pid'])
    if current==record:raise RuntimeError('prior owned process is still alive')
    # A different creation time establishes the recorded process has ended.


def wait_dead(record,seconds):
    """Wait for the recorded interpreter, not merely its virtualenv launcher."""
    if record is None:raise RuntimeError('owned interpreter identity missing')
    if process_identity(record['pid'])!=record:return
    k=kernel();k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
    k.CloseHandle.argtypes=[w.HANDLE];k.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD]
    handle=k.OpenProcess(0x1000|0x00100000,False,record['pid'])
    if not handle:
        if ctypes.get_last_error()==87:return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        k.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.POINTER(w.FILETIME)]*4
        created,ended,system,user=(w.FILETIME() for _ in range(4))
        if not k.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(ended),ctypes.byref(system),ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        if (created.dwHighDateTime<<32)|created.dwLowDateTime!=record['creation_filetime']:return
        if k.WaitForSingleObject(handle,max(0,int(seconds*1000)))!=0:
            raise TimeoutError('owned interpreter termination not proved within closure reserve')
    finally:k.CloseHandle(handle)


class OwnedJob:
    def __init__(self):
        if os.name!='nt':raise RuntimeError('reviewed supervision requires Windows')
        self.k=kernel();self.k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];self.k.CreateJobObjectW.restype=w.HANDLE
        self.k.CloseHandle.argtypes=[w.HANDLE]
        self.handle=self.k.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        class Basic(ctypes.Structure):
            _fields_=[('ProcessTime',ctypes.c_longlong),('JobTime',ctypes.c_longlong),('Flags',w.DWORD),
                ('MinWS',ctypes.c_size_t),('MaxWS',ctypes.c_size_t),('Active',w.DWORD),
                ('Affinity',ctypes.c_size_t),('Priority',w.DWORD),('Scheduling',w.DWORD)]
        class IO(ctypes.Structure):_fields_=[(n,ctypes.c_ulonglong) for n in ('r','w','o','rb','wb','ob')]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_mem',ctypes.c_size_t),('job_mem',ctypes.c_size_t),
                      ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        limits=Extended();limits.basic.Flags=0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        self.k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.WinError(ctypes.get_last_error());self.close();raise error

    def assign(self,child):
        self.assign_handle(w.HANDLE(int(child._handle)))

    def assign_handle(self,handle):
        self.k.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)]
        already=w.BOOL()
        if not self.k.IsProcessInJob(handle,self.handle,ctypes.byref(already)):
            raise ctypes.WinError(ctypes.get_last_error())
        if already.value:return
        self.k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        if not self.k.AssignProcessToJobObject(self.handle,handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_pid(self,pid):
        self.k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];self.k.OpenProcess.restype=w.HANDLE
        handle=self.k.OpenProcess(0x1000|0x0100|0x0001,False,pid)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        try:self.assign_handle(handle)
        finally:self.k.CloseHandle(handle)

    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None


def available_ram():
    class Memory(ctypes.Structure):
        _fields_=[('length',w.DWORD),('load',w.DWORD)]+[(n,ctypes.c_ulonglong) for n in
            ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    state=Memory();state.length=ctypes.sizeof(state)
    if not kernel().GlobalMemoryStatusEx(ctypes.byref(state)):raise ctypes.WinError(ctypes.get_last_error())
    return dict(total_bytes=state.total,available_bytes=state.available)
