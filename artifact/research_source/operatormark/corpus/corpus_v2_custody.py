"""Versioned v2 byte custody. No ACL operations, enrollment, solves or grants.

Physical native qualification is a separate prerequisite. These routines preserve
all interrupted files; a failed staging/publication operation is never retried.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import stat
import zipfile

from . import canonical as C
from .corpus_v2_schema import V2Layout, tensor_inventory, inventory_sha256
from .corpus_format import SPLIT_SAMPLE_COUNTS, SHARD_SIZE


class V2CustodyError(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise V2CustodyError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Population:
    layout: V2Layout
    counts: tuple[int, int, int]
    shard_size: int
    synthetic: bool

    def __post_init__(self):
        need(type(self.layout) is V2Layout and type(self.synthetic) is bool, 'explicit v2 layout/mode required')
        need(type(self.counts) is tuple and len(self.counts) == 3 and
             all(type(n) is int and n >= 0 for n in self.counts), 'population')
        need(type(self.shard_size) is int and self.shard_size > 0 and
             all(n % self.shard_size == 0 for n in self.counts), 'shard population')
        need(self.synthetic or (self.counts == tuple(SPLIT_SAMPLE_COUNTS.values()) and self.shard_size == SHARD_SIZE),
             'production population cannot be reduced')

    @property
    def samples(self):
        return sum(self.counts)

    @property
    def directions(self):
        return self.counts[0] + 2 * sum(self.counts[1:])

    def shards(self):
        return tuple((s, i) for s, count in zip(('training','validation','test'),self.counts)
                     for i in range(count // self.shard_size))

    def record(self):
        return dict(inventory_sha256=inventory_sha256(self.layout), n_ref=self.layout.n_ref,
                    retention=self.layout.retention, counts=list(self.counts), shard_size=self.shard_size,
                    samples=self.samples, directions=self.directions, synthetic=self.synthetic)


def file_population(population):
    """Independent of writer manifests, including direction/native populations."""
    need(type(population) is Population, 'exact population required')
    rows = []
    for split, shard in population.shards():
        directory = 'test.sealed' if split == 'test' else split
        base = f'corpus_staging/{directory}/shard_{shard:04d}/'
        for family in tensor_inventory(population.layout):
            if family.stored:
                count = population.shard_size * family.items_per_sample(split)
                rows.append((base+family.name+'.float64.npy', 'tensor', (count,1,family.side,family.side)))
        rows.extend((base+name, kind, ()) for name,kind in (
            ('verification.jsonl','samples'),('directions.jsonl','directions'),('shard_manifest.json','manifest')))
    rows.extend((name,'record',()) for name in ('corpus_configuration.json','failures.jsonl',
                 'unwritten_shard_measurements.jsonl','generation_summary.json','run/generation_run.json'))
    return tuple(rows)


def relative(name):
    from .state_provider_bindings import relative as validate
    validate(name)
    return name


@dataclass(frozen=True)
class FileFact:
    name: str
    size: int
    sha256: str
    device: int
    inode: int

    def content(self):
        return dict(name=self.name,size=self.size,sha256=self.sha256)


def _identity(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _cross_api_identity(s):
    # Windows 3.13.5 path stat exposes birth time as ctime, while descriptor
    # fstat exposes change time after rename. Compare birth explicitly across
    # APIs; retain ctime checks independently within each API below.
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,getattr(s,'st_birthtime_ns',None))


def checked_directory(path):
    path=Path(path).absolute()
    for value in (path,*path.parents):
        s=value.lstat()
        need(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode) and
             not getattr(s,'st_file_attributes',0)&0x400,'ordinary non-reparse directory required')
    return path


def acquire_file(root, name):
    """Streaming independent bytes/hash; reject aliases and mutation during read."""
    root=checked_directory(root);relative(name);path=root/name
    for ancestor in (path,*path.parents):
        if ancestor == root:break
        value=ancestor.lstat()
        need(not stat.S_ISLNK(value.st_mode) and not getattr(value,'st_file_attributes',0)&0x400,
             'reparse/symlink in custody path')
    before=path.stat();need(stat.S_ISREG(before.st_mode), 'regular custody file required')
    h=hashlib.sha256();count=0
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno());need(_cross_api_identity(opened)==_cross_api_identity(before),'open substitution')
        while block:=stream.read(1024*1024):h.update(block);count+=len(block)
        need(_identity(os.fstat(stream.fileno()))==_identity(opened),'mutation during read')
    after=path.stat()
    need(_identity(after)==_identity(before) and count==before.st_size,'source mutation during acquisition')
    return FileFact(name,count,h.hexdigest(),before.st_dev,before.st_ino)


def names(root):
    root=checked_directory(root)
    result=[]
    for directory, dirs, files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(directory)/name;s=p.lstat()
            need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400,
                 'reparse entry in retained inventory')
        result.extend((Path(directory)/name).relative_to(root).as_posix() for name in files)
    return tuple(sorted(result))


def snapshot(root, expected_names):
    expected=tuple(sorted(expected_names))
    need(len(expected)==len(set(expected)), 'duplicate expected member')
    before=names(root)
    need(before==expected, f'retained membership differs: added={sorted(set(before)-set(expected))}; missing={sorted(set(expected)-set(before))}')
    rows=tuple(acquire_file(root,name) for name in expected)
    need(names(root)==expected,'final retained membership changed')
    return rows


def same_content(left,right,*,identities=True):
    need(tuple(x.content() for x in left)==tuple(x.content() for x in right),'retained content/membership changed')
    if identities:
        need(tuple((x.device,x.inode) for x in left)==tuple((x.device,x.inode) for x in right), 'custody identity changed')


def tree(root):
    """One walk: (every directory, including empty ones; every non-directory entry)."""
    root=checked_directory(root)
    found=([],[])
    for directory, dirs, files in os.walk(root,followlinks=False):
        for index, group in enumerate((dirs, files)):
            for name in group:
                p=Path(directory)/name;s=p.lstat()
                need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&0x400,
                     'reparse entry in retained inventory')
                found[index].append(p.relative_to(root).as_posix())
    return tuple(sorted(found[0])),tuple(sorted(found[1]))


def directories(root):
    """Every directory below root, including empty ones; names() lists files only."""
    return tree(root)[0]


def ancestor_directories(names):
    return tuple(sorted({'/'.join(Path(n).parts[:i]) for n in names for i in range(1,len(Path(n).parts))}))


def exact_generated_population(root,population,*,ladders=False):
    """Exact file AND directory membership: an empty foreign directory is refused.

    The final check re-walks files and directories together, after the content
    snapshot, so nothing created during acquisition is missed.
    """
    required=[r[0] for r in file_population(population)]
    if ladders:required.append('anomaly_ladders.jsonl')
    expected=(ancestor_directories(required),tuple(sorted(required)))
    def check(final):
        actual=tree(root)
        for label,have,want in tuple(zip(('directory','file'),actual,expected))[:1+final]:
            need(have==want,f'retained {label} membership differs: added={sorted(set(have)-set(want))}; '
                 f'missing={sorted(set(want)-set(have))}')
    check(False);rows=snapshot(root,required);check(True)
    return rows


def exact_tree(root,names):
    """Exactly these files and exactly their ancestor directories, nothing else
    (no empty foreign directory). Same two-walk pattern as the payload check."""
    expected=(ancestor_directories(names),tuple(sorted(names)))
    def check(final):
        actual=tree(root)
        for label,have,want in tuple(zip(('directory','file'),actual,expected))[:1+final]:
            need(have==want,f'retained record-root {label} membership differs: added={sorted(set(have)-set(want))}; '
                 f'missing={sorted(set(want)-set(have))}')
    check(False);rows=snapshot(root,names);check(True)
    return rows


def exact_record_root(root,names):
    """The flat evidence root: exactly these files and no directory at all."""
    need(all('/' not in n for n in names),'flat record root expected')
    return exact_tree(root,names)


def write_record(root,name,body):
    relative(name);root=checked_directory(root);path=root/name
    current=root
    for part in Path(name).parts[:-1]:
        current=current/part
        if current.exists():checked_directory(current)
        else:current.mkdir()
    path.parent.mkdir(parents=True,exist_ok=True)
    data=C.canonical_json_bytes(C.with_self_hash(body,'record_self_sha256'))
    C.write_canonical_json_exclusive(path,C.strict_canonical_json_loads(data))
    fact=acquire_file(root,name)
    need(fact.sha256==digest(data),'record independent readback')
    return fact


def promote_once(staging,final,expected):
    staging=Path(staging);final=Path(final)
    checked_directory(staging.parent);checked_directory(final.parent)
    need(staging.absolute()!=final.absolute(),'distinct whole-attempt destinations required')
    need(staging.is_dir() and not final.exists(),'exclusive whole-attempt promotion')
    original=staging.stat()
    need(original.st_dev==final.parent.stat().st_dev,'same-volume promotion required')
    same_content(expected,snapshot(staging,[r.name for r in expected]))
    os.rename(staging,final)  # no copy/delete fallback and no retry
    moved=final.stat()
    need((original.st_dev,original.st_ino)==(moved.st_dev,moved.st_ino),'moving-directory identity differs')
    after=snapshot(final,[r.name for r in expected])
    same_content(expected,after)
    return after


def classic_archive_size(members):
    """Exact deterministic ZIP_STORED framing, no extras/comments/data descriptors."""
    rows=tuple(members)
    need(len(rows)<=65535 and len({n for n,_ in rows})==len(rows),'classic ZIP entry domain')
    total=22
    for name,size in rows:
        relative(name);need(name.isascii() and type(size) is int and 0<=size<2**31,'classic ZIP member domain')
        total+=76+2*len(name.encode('ascii'))+size
    need(total<2**31,'Python classic ZIP offset domain; ZIP64 policy not adopted')
    return total


def archive_once(evidence_root,destination,expected):
    """Stream, preserve partial ZIP, and independently read every member back."""
    evidence_root=Path(evidence_root);destination=Path(destination)
    need(not destination.exists(),'archive destination exists')
    partial=destination.with_name(destination.name+'.__creating__')
    need(not partial.exists(),'partial archive already exists')
    same_content(expected,snapshot(evidence_root,[r.name for r in expected]))
    size=classic_archive_size((r.name,r.size) for r in expected)
    with partial.open('xb') as output:
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED,allowZip64=False) as archive:
            for row in expected:
                info=zipfile.ZipInfo(row.name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
                with (evidence_root/row.name).open('rb') as source, archive.open(info,'w') as target:
                    while block:=source.read(1024*1024):target.write(block)
        output.flush();os.fsync(output.fileno())
    need(partial.stat().st_size==size,'archive framing/accounting mismatch')
    verify_archive(partial,expected)
    same_content(expected,snapshot(evidence_root,[r.name for r in expected]))
    os.rename(partial,destination)
    verify_archive(destination,expected)
    return acquire_file(destination.parent,destination.name)


def verify_archive(path,expected):
    before=Path(path).stat()
    with zipfile.ZipFile(path,'r') as archive:
        need(tuple(archive.namelist())==tuple(r.name for r in expected),'archive exact ordered membership')
        for info,row in zip(archive.infolist(),expected,strict=True):
            need(info.file_size==row.size and info.compress_type==zipfile.ZIP_STORED and not info.extra and not info.comment,
                 'archive metadata mismatch')
            h=hashlib.sha256();count=0
            with archive.open(info) as source:
                while block:=source.read(1024*1024):h.update(block);count+=len(block)
            need(count==row.size and h.hexdigest()==row.sha256,'independent archive member readback')
    need(_identity(Path(path).stat())==_identity(before),'archive mutated during readback')
