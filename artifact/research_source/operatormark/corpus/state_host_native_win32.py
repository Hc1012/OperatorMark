"""Existing E3 held-handle Windows primitives; copied unchanged for State host reuse.
Source: artifacts/e3_proposal_20260921/harness_source_e3/owned_process.py::NativeBindings.
No native calls at import. Does not issue execution authority."""
class NativeBindings:
    def __init__(self):
        # Only the external supervised harness constructs this object. Candidate
        # tests inject a fake instead; importing this module does not load a DLL.
        import ctypes as c
        from ctypes import wintypes as w
        import _winapi
        import msvcrt
        import os
        import subprocess
        self.c, self.w, self.winapi = c, w, _winapi
        self.msvcrt, self.os, self.subprocess = msvcrt, os, subprocess
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        size = c.c_size_t
        class Limits(c.Structure):
            _fields_ = [('process_time', c.c_longlong), ('job_time', c.c_longlong), ('flags', w.DWORD),
                ('min_ws', size), ('max_ws', size), ('active_limit', w.DWORD), ('affinity', size),
                ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(n, c.c_ulonglong) for n in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
        class Extended(c.Structure):
            _fields_ = [('basic', Limits), ('io', IO), ('process_mem', size), ('job_mem', size), ('peak_process', size), ('peak_job', size)]
        class Members(c.Structure):
            _fields_ = [('assigned', w.DWORD), ('count', w.DWORD), ('pids', size * 8)]
        self.Extended, self.Members = Extended, Members
        self.create_job = self.api('CreateJobObjectW', [c.c_void_p, w.LPCWSTR], w.HANDLE)
        self.set_job = self.api('SetInformationJobObject', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL)
        self.assign = self.api('AssignProcessToJobObject', [w.HANDLE, w.HANDLE], w.BOOL)
        self.resume_thread = self.api('ResumeThread', [w.HANDLE], w.DWORD)
        self.query = self.api('QueryInformationJobObject', [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL)
        self.job_terminate = self.api('TerminateJobObject', [w.HANDLE, w.UINT], w.BOOL)
        self.is_in_job = self.api('IsProcessInJob', [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)], w.BOOL)
        self.times = self.api('GetProcessTimes', [w.HANDLE, c.POINTER(w.FILETIME), c.POINTER(w.FILETIME), c.POINTER(w.FILETIME), c.POINTER(w.FILETIME)], w.BOOL)
        self.pid_of = self.api('GetProcessId', [w.HANDLE], w.DWORD)
        self.image_of = self.api('QueryFullProcessImageNameW', [w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)], w.BOOL)

    def api(self, name, args, result):
        f = getattr(self.kernel, name)
        f.argtypes, f.restype = args, result
        return f

    def check(self, value):
        if not value:
            raise self.c.WinError(self.c.get_last_error())
        return value

    def create_private_job(self):
        handle = self.check(self.create_job(None, None))
        limits = self.Extended()
        limits.basic.flags = 0x2000
        self.check(self.set_job(handle, 9, self.c.byref(limits), self.c.sizeof(limits)))
        return handle

    def create_suspended(self, command, cwd, environment, stdin, stdout, stderr):
        info = self.subprocess.STARTUPINFO()
        info.dwFlags |= self.subprocess.STARTF_USESTDHANDLES
        handles = [self.msvcrt.get_osfhandle(f.fileno()) for f in (stdin, stdout, stderr)]
        info.hStdInput, info.hStdOutput, info.hStdError = handles
        # E3's creation-held single-process jobs need no runtime transfer.
        # Keep the inherited receiver path only for the retained E2 interface.
        if 'E2_HANDLE_RECEIVER' in environment:
            handles.append(int(environment['E2_HANDLE_RECEIVER']))
        info.lpAttributeList = {'handle_list': handles}
        try:
            for handle in handles:
                self.os.set_handle_inheritable(handle, True)
            ph, th, pid, _ = self.winapi.CreateProcess(command[0], self.subprocess.list2cmdline(command),
                None, None, True, 0x4 | 0x400, environment, str(cwd), info)
            return ph, th, pid
        finally:
            for handle in handles:
                self.os.set_handle_inheritable(handle, False)

    def assign_job(self, job, process):
        self.check(self.assign(job, process))

    def resume(self, thread):
        if self.resume_thread(thread) == 0xffffffff:
            raise self.c.WinError(self.c.get_last_error())

    def describe(self, handle, role):
        w, c = self.w, self.c
        creation, exit_time, kernel, user = w.FILETIME(), w.FILETIME(), w.FILETIME(), w.FILETIME()
        self.check(self.times(handle, c.byref(creation), c.byref(exit_time), c.byref(kernel), c.byref(user)))
        return dict(pid=self.check(self.pid_of(handle)), role=role,
                    creation_filetime=(creation.dwHighDateTime << 32) | creation.dwLowDateTime,
                    exit_code=self.exit_code(handle) if self.signaled(handle) else None)

    def executable(self, handle):
        buf = self.c.create_unicode_buffer(32768)
        count = self.w.DWORD(len(buf))
        self.check(self.image_of(handle, 0, buf, self.c.byref(count)))
        return buf.value

    def observation(self, job):
        value = self.Members()
        members_ok = bool(self.query(job, 3, self.c.byref(value), self.c.sizeof(value), None))
        members_error = self.c.get_last_error() if not members_ok else None
        # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION: four LARGE_INTEGER fields,
        # followed by four DWORD counters; TotalProcesses is at byte 36.
        accounting = (self.c.c_uint64 * 6)()
        accounting_ok = bool(self.query(job, 1, self.c.byref(accounting), 48, None))
        accounting_error = self.c.get_last_error() if not accounting_ok else None
        counters = self.c.cast(self.c.byref(accounting,32),self.c.POINTER(self.w.DWORD))
        return dict(members_query_ok=members_ok, members_error=members_error,
            accounting_query_ok=accounting_ok, accounting_error=accounting_error,
            members_raw=bytes(value).hex(), accounting_raw=bytes(accounting).hex(),
            assigned=value.assigned, listed=value.count, members=list(value.pids[:min(value.count,8)]),
            total_page_fault_count=counters[0], total_processes=counters[1],
            active_processes=counters[2], limit_terminated_processes=counters[3])

    def in_job(self, handle, job):
        result = self.w.BOOL()
        self.check(self.is_in_job(handle, job, self.c.byref(result)))
        return bool(result.value)

    def signaled(self, handle):
        return self.winapi.WaitForSingleObject(handle, 0) == 0

    def wait(self, handle, remaining):
        return self.winapi.WaitForSingleObject(handle, max(0, int(remaining * 1000))) == 0

    def exit_code(self, handle):
        return self.winapi.GetExitCodeProcess(handle)

    def terminate_job(self, handle, code):
        self.check(self.job_terminate(handle, code))

    def terminate_process(self, handle, code):
        self.winapi.TerminateProcess(handle, code)

    def close_handle(self, handle):
        self.winapi.CloseHandle(handle)


