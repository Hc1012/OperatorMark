"""Adopted B: bounded retained bytes, observed-only hashes, explicit loss.

EOF means the reader actually observed end of pipe, not that all bytes were
retained. No claim about unobserved pipe contents or pipe capacity is made.
"""
import hashlib
from pathlib import Path

QUOTA_BYTES = 64 * 2**20
MAX_CHUNKS = 4096
MAX_READ = 65536
MAX_COUNTER = 2**63-1


class StreamRetention:
    def __init__(self, sink, *, quota=QUOTA_BYTES):
        if type(quota) is not int or not 0 < quota <= QUOTA_BYTES:
            raise ValueError('retention quota')
        self.sink,self.quota=sink,quota
        self.observed=self.retained=self.discarded=0
        self.observed_hash=hashlib.sha256();self.retained_hash=hashlib.sha256()
        self.eof=False;self.overflow=False;self.write_failed=False
        self.count_exact=True

    def feed(self,data,*,metadata_available=True):
        if type(data) is not bytes or not 0<len(data)<=MAX_READ:raise ValueError('bounded raw read')
        actual=self.observed+len(data)
        self.count_exact &= actual<=MAX_COUNTER
        self.observed=min(actual,MAX_COUNTER);self.observed_hash.update(data)
        if not self.count_exact:self.overflow=True
        remaining=max(0,self.quota-self.retained) if metadata_available else 0
        keep=data[:remaining];self.overflow |= len(keep)!=len(data)
        committed=0
        try:
            while committed<len(keep):
                n=self.sink.write(memoryview(keep)[committed:])
                if n is None or not 0<n<=len(keep)-committed:raise OSError('short stream write')
                self.retained_hash.update(keep[committed:committed+n]);self.retained+=n;committed+=n
        except BaseException:
            self.write_failed=True;self.overflow=True;raise
        finally:self.discarded=min(MAX_COUNTER,self.discarded+len(data)-committed)
        return keep

    def record(self):
        return dict(quota_bytes=self.quota,observed_bytes=self.observed if self.count_exact else None,
            observed_bytes_lower_bound=self.observed,observed_count_exact=self.count_exact,
            observed_sha256=self.observed_hash.hexdigest(),retained_bytes=self.retained,
            retained_sha256=self.retained_hash.hexdigest(),discarded_observed_bytes=self.discarded if self.count_exact else None,
            eof_observed=self.eof,capture_complete=self.eof and not self.overflow and not self.write_failed,
            unobserved_bytes=None,overflow=self.overflow,write_failed=self.write_failed)


def verify_retained(record,path):
    """Independent streaming readback of retained bytes; never of discarded data."""
    if not 0<=record['retained_bytes']<=record['quota_bytes']<=QUOTA_BYTES:
        raise ValueError('retained stream bound')
    size=0;digest=hashlib.sha256();path=Path(path)
    before=path.stat()
    with path.open('rb') as source:
        while data:=source.read(MAX_READ):size+=len(data);digest.update(data)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('retained stream changed')
    if size!=record['retained_bytes'] or digest.hexdigest()!=record['retained_sha256']:
        raise ValueError('retained stream readback differs')
    if record['capture_complete']:
        if (not record['eof_observed'] or record['overflow'] or record['write_failed'] or
            not record['observed_count_exact'] or record['discarded_observed_bytes']!=0 or
            record['observed_bytes']!=size or record['observed_sha256']!=digest.hexdigest()):
            raise ValueError('complete stream claim inconsistent')
    return dict(retained_bytes=size,retained_sha256=digest.hexdigest(),readback_verified=True,
        discarded_bytes_independently_verified=False)
