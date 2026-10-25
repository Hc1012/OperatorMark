"""Single-writer raw update stream beside existing branch checkpoints.

Uses the project's append/flush/fsync pattern (q6_common.event), without its
per-row console replay. No resume, acceptance, reservation or retry mechanism.
One JSON record is buffered at a time; incomplete bytes are never truncated.
"""
import hashlib
import json
import os
import math
import sys
from pathlib import Path

MAX_RECORD_BYTES = 65536

# CPython binary64 JSON repr: <=17 significant digits, optional sign/dot,
# 'e', exponent sign and <=3 exponent digits; fixed notation is shorter.
# This is a source/runtime-scoped producer bound, NOT a new reader ceiling.
FINITE_FLOAT_BYTES = 24
NORMAL_RECIPE = {'D9','forward32','sham','A4','A5B2','A5B3'}


def normal_bound(nodes):
    """Admit the current Engine row producer and price exact serialized IDs.

    Attempt/completion integers are nonnegative and bounded by the finite plan.
    Owner IDs, paths, tensor values and exception text are not in normal rows.
    json.dumps defaults ensure_ascii=True, so the exact quoted node length below
    includes all escaping (including paired surrogate escapes for non-BMP text).
    """
    if (sys.implementation.name!='cpython' or (sys.float_info.radix,sys.float_info.mant_dig,sys.float_info.max_exp)!=(2,53,1024)):
        raise ValueError('normal-journal binary64 serializer scope required')
    if not isinstance(nodes,list) or not nodes:raise ValueError('finite normal-journal plan required')
    ids=set();total=0;maximum=0
    for n in nodes:
        if (not isinstance(n,dict) or not isinstance(n.get('id'),str) or not n['id'] or n['id'] in ids
            or n.get('recipe') not in NORMAL_RECIPE
            or type(n.get('start')) is not int or type(n.get('end')) is not int
            or not 0<=n['start']<n['end'] or type(n.get('cooldown')) is not bool):
            raise ValueError('normal-journal producer/plan invariant')
        ids.add(n['id']);total+=n['end']-n['start'];maximum=max(maximum,n['end'])
    digits=len(str(total));state_digits=len(str(maximum));rows=[]
    def size(v):return len((json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8'))
    for n in nodes:
        attempt=size(dict(event='attempt',attempt=0,node=n['id'],pre_update=0,lr=0.0))+digits-1+state_digits-1+FINITE_FLOAT_BYTES-3
        completion=size(dict(event='completed',attempt=0,node=n['id'],executed_update=0,
            row=dict(update=0,loss=0.0,lr=0.0)))+2*(digits-1)+state_digits-1+2*(FINITE_FLOAT_BYTES-3)
        if max(attempt,completion)>MAX_RECORD_BYTES:raise ValueError('normal plan exceeds generic journal ceiling')
        rows.append(dict(node=n['id'],start=n['start'],end=n['end'],updates=n['end']-n['start'],
            attempt_bytes=attempt,completion_bytes=completion))
    return dict(producer='msdev.kernels.Engine.step.v1',updates=total,records=2*total,
        integer_digits=digits,state_integer_digits=state_digits,nodes=rows,
        attempt_max_bytes=max(r['attempt_bytes'] for r in rows),completion_max_bytes=max(r['completion_bytes'] for r in rows),
        bytes_upper=sum(r['updates']*(r['attempt_bytes']+r['completion_bytes']) for r in rows),
        generic_reader_ceiling=MAX_RECORD_BYTES)


class UpdateJournal:
    def __init__(self, path, *, normal_nodes=None):
        self.normal=None if normal_nodes is None else normal_bound(normal_nodes)
        self.path=Path(path)
        with self.path.open('xb') as f:f.flush();os.fsync(f.fileno())
        self.offset=0;self.digest=hashlib.sha256();self.attempts=0;self.completed=0
        self.pending=None;self.poisoned=False;self.uncommitted_record=None

    def _append(self, value):
        if self.poisoned:raise RuntimeError('journal failed; further writes refused')
        self.uncommitted_record=value
        raw=(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
        if len(raw)>MAX_RECORD_BYTES:raise ValueError('update journal record exceeds fixed record domain')
        try:
            # Open only this owned fresh stream. No replace-in-place live file.
            with self.path.open('ab') as f:
                if f.tell()!=self.offset:raise ValueError('update journal append position drift')
                if f.write(raw)!=len(raw):raise OSError('short update journal write')
                f.flush();os.fsync(f.fileno())
        except BaseException:
            self.poisoned=True
            raise
        self.digest.update(raw);self.offset+=len(raw);self.uncommitted_record=None

    def begin(self,node,pre_update,lr):
        if self.pending is not None:raise ValueError('unfinished update attempt')
        if self.normal is not None:
            planned=next((r for r in self.normal['nodes'] if r['node']==node),None)
            if (planned is None or type(pre_update) is not int or not planned['start']<=pre_update<planned['end']
                or type(lr) is not float or not math.isfinite(lr) or self.attempts>=self.normal['updates']):
                raise ValueError('normal-journal admission before optimizer dispatch')
        value=dict(event='attempt',attempt=self.attempts+1,node=node,pre_update=pre_update,lr=lr)
        self._append(value)
        self.attempts+=1;self.pending=value

    def finish(self,row):
        if self.pending is None:raise ValueError('completion without attempt')
        if row.get('update')!=self.pending['pre_update']+1:
            raise ValueError('completed update state differs')
        value=dict(event='completed',attempt=self.attempts,node=self.pending['node'],
                   executed_update=self.completed+1,row=row)
        self._append(value);self.completed+=1;self.pending=None
        if self.normal is not None and (set(row)!={'update','loss','lr'} or type(row['update']) is not int
            or any(type(row[k]) is not float or not math.isfinite(row[k]) for k in ('loss','lr'))):
            # Do not discard an unexpected producer row. Its original bytes are
            # already durable under the unchanged generic ceiling; fail the run
            # and account this abnormal/failure path outside the normal estimate.
            raise ValueError('normal-journal row schema differs; original row retained')

    def snapshot(self):
        return dict(path=self.path.name,acknowledged_bytes=self.offset,sha256=self.digest.hexdigest(),
                    attempted_updates=self.attempts,completed_updates=self.completed,
                    pending_attempt=None if self.pending is None else self.pending['attempt'],
                    write_failed=self.poisoned,uncommitted_record=self.uncommitted_record)


def read_prefix(path, on_record=None):
    """Stream actual bytes; identify rather than repair an incomplete tail.

    A syntactically complete readback prefix is not a historical fsync receipt.
    Compare its offset/hash with saved checkpoint/final acknowledged anchors.
    A pending intent cannot establish whether the operation ran before a crash.
    """
    path=Path(path);digest=hashlib.sha256();offset=0;attempts=completed=0;pending=None;error=None
    with path.open('rb') as f:
        while True:
            raw=f.readline(MAX_RECORD_BYTES+1)
            if not raw:break
            try:
                if len(raw)>MAX_RECORD_BYTES or not raw.endswith(b'\n'):raise ValueError('incomplete/oversized record')
                value=json.loads(raw)
                if value.get('event')=='attempt':
                    if pending is not None or value.get('attempt')!=attempts+1:raise ValueError('attempt order')
                    if type(value.get('pre_update')) is not int or value['pre_update']<0:raise ValueError('pre-update state')
                    next_pending=value
                elif value.get('event')=='completed':
                    if (pending is None or value.get('attempt')!=attempts or value.get('node')!=pending['node']
                            or value.get('executed_update')!=completed+1
                            or value.get('row',{}).get('update')!=pending['pre_update']+1):
                        raise ValueError('completion order/state')
                    next_pending=None
                else:raise ValueError('unknown update event')
            except (ValueError,TypeError,KeyError,UnicodeError) as exc:
                error=str(exc);break
            if value['event']=='attempt':attempts+=1
            else:completed+=1
            pending=next_pending;digest.update(raw);offset+=len(raw)
            if on_record is not None:on_record(value)
    return dict(path=path.name,verified_prefix_bytes=offset,sha256=digest.hexdigest(),
                attempted_updates=attempts,completed_updates=completed,
                pending_attempt=None if pending is None else pending['attempt'],
                trailing_bytes=path.stat().st_size-offset,error=error,
                complete=error is None and pending is None)
