"""Non-numerical persistence regression; no model or solver imports."""
import ctypes
import os
from pathlib import Path
import threading
import time
import pytest
import core


def denied(code=5):
    error=PermissionError('injected Windows sharing/access denial')
    error.winerror=code
    return error


def budget(tmp_path):
    return core.Budget(tmp_path/'COUNTERS.json',dict(solver_attempts=10),time.time()+10)


def test_transient_same_pending_exactly_once(tmp_path,monkeypatch):
    b=budget(tmp_path);original=os.replace;pending=[];operations=[]
    def replace(src,dst):
        pending.append((str(src),Path(src).read_bytes()))
        if len(pending)<3:raise denied(32)
        original(src,dst)
    monkeypatch.setattr(core.os,'replace',replace)
    b.charge('solver_attempts');operations.append('operation')
    assert len(pending)==3 and len(set(pending))==1
    assert b.counts['solver_attempts']==1
    assert core.read(b.path)['counts']['solver_attempts']==1
    assert operations==['operation'] and not b.path.with_suffix('.next').exists()


def test_persistent_no_operation_no_recharge_pending_retained(tmp_path,monkeypatch):
    b=budget(tmp_path);calls=[];operations=[]
    def replace(src,dst):
        calls.append(Path(src).read_bytes());raise denied()
    monkeypatch.setattr(core.os,'replace',replace)
    start=time.monotonic()
    with pytest.raises(PermissionError):
        b.charge('solver_attempts');operations.append('forbidden')
    assert time.monotonic()-start<1
    assert len(calls)==6 and len(set(calls))==1
    assert core.read(b.path)['counts']['solver_attempts']==0
    assert core.read(b.path.with_suffix('.next'))['counts']['solver_attempts']==1
    with pytest.raises(RuntimeError,match='unresolved persistence'):
        b.charge('solver_attempts');operations.append('forbidden')
    assert not operations and b.counts['solver_attempts']==1 and len(calls)==6


def test_deadline_during_retry_preserves_pending(tmp_path,monkeypatch):
    b=budget(tmp_path);b.deadline=time.time()+.015;calls=[]
    def replace(src,dst):calls.append(1);raise denied()
    monkeypatch.setattr(core.os,'replace',replace)
    with pytest.raises((TimeoutError,PermissionError)):b.charge('solver_attempts')
    assert 1<=len(calls)<6 and b.persistence_failed
    assert b.path.with_suffix('.next').exists()


def test_unrelated_error_not_retried_and_existing_pending_preserved(tmp_path,monkeypatch):
    path=tmp_path/'record.json';calls=[]
    def replace(src,dst):calls.append(1);raise OSError('not a sharing error')
    monkeypatch.setattr(core.os,'replace',replace)
    with pytest.raises(OSError):core.replace_record(path,{'v':1})
    saved=path.with_suffix('.next').read_bytes()
    with pytest.raises(FileExistsError):core.replace_record(path,{'v':2})
    assert calls==[1] and path.with_suffix('.next').read_bytes()==saved


@pytest.mark.skipif(os.name!='nt',reason='actual Windows file sharing contract')
def test_windows_reader_release_allows_same_pending_replacement(tmp_path,monkeypatch):
    from ctypes import wintypes
    b=budget(tmp_path);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    # Temporary fixture only: allow read/write sharing, deliberately deny delete sharing.
    handle=kernel.CreateFileW(str(b.path),0x80000000,3,None,3,0,None)
    assert handle!=ctypes.c_void_p(-1).value
    original=os.replace;failures=[];pending=[];release=threading.Event()
    def replace(src,dst):
        pending.append(Path(src).read_bytes())
        try:return original(src,dst)
        except OSError as exc:
            failures.append(exc.winerror);release.set();raise
    def close_reader():
        release.wait(1);kernel.CloseHandle(handle)
    thread=threading.Thread(target=close_reader);thread.start()
    monkeypatch.setattr(core.os,'replace',replace)
    try:b.charge('solver_attempts')
    finally:release.set();thread.join(timeout=2)
    assert not thread.is_alive() and failures and all(x in (5,32,33) for x in failures)
    assert len(set(pending))==1 and core.read(b.path)['counts']['solver_attempts']==1
