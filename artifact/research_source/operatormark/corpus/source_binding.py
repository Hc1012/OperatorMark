"""Private in-memory source correspondence, not authentication or a disk schema.

The caller must independently accept the expectations and bootstrap/interpreter/
dependency premises. This module checks supplied bytes; it never acquires files,
discovers trusted membership, authorizes a state, or establishes live custody.
"""
from dataclasses import dataclass
import hashlib
import keyword
import re
import unicodedata


class SourceBindingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceMember:
    path: str
    module: str | None
    is_package: bool
    file_bytes: int
    file_sha256: str
    git_blob: str


@dataclass(frozen=True, slots=True)
class TrustPremise:
    role: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ExecutionSourceExpectations:
    operation: str
    role: str
    protocol_sha256: str
    implementation_commit: str
    implementation_tree: str
    project_roots: tuple[str, ...]
    members: tuple[SourceMember, ...]
    entry_point: str
    permitted_imports: tuple[tuple[str, str], ...]
    external_imports: tuple[str, ...]
    trust_premises: tuple[TrustPremise, ...]


@dataclass(frozen=True, slots=True)
class SourceBytes:
    path: str
    data: bytes


@dataclass(frozen=True, slots=True)
class VerifiedSourceSet:
    """A value, never a capability. Consumers must reverify against expectations."""
    expected: ExecutionSourceExpectations
    sources: tuple[SourceBytes, ...]


def _need(condition, reason):
    if not condition:
        raise SourceBindingError(reason)


def _text(value):
    _need(type(value) is str and bool(value), 'nonempty exact text required')
    _need(value == unicodedata.normalize('NFC', value) and
          not any(unicodedata.category(c) in ('Cc', 'Cs') for c in value), 'noncanonical text')


def _hash(value, width=64):
    _need(type(value) is str and re.fullmatch('[0-9a-f]{'+str(width)+'}', value) is not None
          and value != '0'*width, 'invalid source identity')


def _length(value):
    _need(type(value) is int and value >= 0, 'exact nonnegative byte length required')


def _module(value):
    _text(value)
    _need(all(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', p) and not keyword.iskeyword(p)
              for p in value.split('.')), 'invalid module name')


def _path(value):
    _text(value)
    reserved = {'con', 'prn', 'aux', 'nul', 'clock$',
                *(f'com{i}' for i in '123456789¹²³'), *(f'lpt{i}' for i in '123456789¹²³')}
    _need(not any(c in value for c in '\\:<>"|?*'), 'invalid source path')
    _need(all(p not in ('', '.', '..') and p[-1] not in ' .' and
              p.split('.')[0].casefold() not in reserved for p in value.split('/')), 'aliased source path')


def validate_execution_source_manifest(manifest, *, expected):
    """Validate exact independent inventory/edges, then compare the supplied table."""
    _need(type(expected) is ExecutionSourceExpectations, 'independent expectations required')
    for value in (expected.operation, expected.role):
        _text(value)
    _hash(expected.protocol_sha256)
    _hash(expected.implementation_commit, 40)
    _hash(expected.implementation_tree, 40)
    _need(type(expected.project_roots) is tuple and bool(expected.project_roots), 'project roots required')
    for root in expected.project_roots:
        _module(root)
        _need('.' not in root, 'project roots must be top-level namespaces')
    _need(expected.project_roots == tuple(sorted(set(expected.project_roots))), 'duplicate/unordered project roots')
    _need(type(expected.members) is tuple and bool(expected.members), 'complete immutable inventory required')
    paths, modules = [], {}
    for member in expected.members:
        _need(type(member) is SourceMember, 'exact member type required')
        _path(member.path); _length(member.file_bytes); _hash(member.file_sha256); _hash(member.git_blob, 40)
        _need(type(member.is_package) is bool, 'exact package flag required')
        paths.append(member.path)
        if member.module is None:
            _need(not member.is_package and not member.path.endswith(('.py', '.pyc', '.pyd')),
                  'executable member cannot be opaque')
            continue
        _module(member.module)
        _need(member.module.split('.')[0] in expected.project_roots, 'module outside controlled roots')
        location = member.module.replace('.', '/') + ('/__init__.py' if member.is_package else '.py')
        _need(member.path == location, 'module/path mapping substitution')
        _need(member.module not in modules, 'duplicate module')
        modules[member.module] = member
    _need(paths == sorted(set(paths)) and len({p.casefold() for p in paths}) == len(paths), 'duplicate/aliased/unordered paths')
    _need(len({m.casefold() for m in modules}) == len(modules), 'aliased module names')
    for root in expected.project_roots:
        _need(root in modules and modules[root].is_package, 'omitted root initializer')
    for name in modules:
        parts = name.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            _need(parent in modules and modules[parent].is_package, 'omitted package initializer')
    _module(expected.entry_point)
    _need(expected.entry_point in modules, 'unbound entry point')
    _need(type(expected.permitted_imports) is tuple, 'immutable import edges required')
    for edge in expected.permitted_imports:
        _need(type(edge) is tuple and len(edge) == 2 and all(type(x) is str for x in edge), 'invalid import edge')
        _need(edge[0] in modules and edge[1] in modules, 'unbound import edge')
    _need(expected.permitted_imports == tuple(sorted(set(expected.permitted_imports))), 'duplicate/unordered import edges')
    _need(type(expected.external_imports) is tuple, 'immutable external inventory required')
    for name in expected.external_imports:
        _module(name)
        _need(name.split('.')[0] not in expected.project_roots, 'project cache exemption forbidden')
    _need(expected.external_imports == tuple(sorted(set(expected.external_imports))), 'duplicate/unordered external imports')
    _need(type(expected.trust_premises) is tuple and len(expected.trust_premises) == 3, 'three explicit trust premises required')
    for premise in expected.trust_premises:
        _need(type(premise) is TrustPremise, 'exact trust premise type required')
        _length(premise.file_bytes); _hash(premise.file_sha256)
    _need(tuple(p.role for p in expected.trust_premises) == ('bootstrap', 'dependencies', 'interpreter'), 'trust premise roles mismatch')
    _need(type(manifest) is tuple and all(type(m) is SourceMember for m in manifest), 'immutable supplied manifest required')
    _need(manifest == expected.members, 'supplied inventory differs from independent expectations')


def verify_execution_source_bytes(*, expected, supplied, manifest=None):
    """Recompute every file and Git-blob identity, including opaque members."""
    validate_execution_source_manifest(expected.members if manifest is None else manifest, expected=expected)
    _need(type(supplied) is tuple and all(type(row) is SourceBytes for row in supplied), 'immutable source acquisitions required')
    for row in supplied:
        _path(row.path)
    _need(tuple(row.path for row in supplied) == tuple(m.path for m in expected.members), 'missing/extra/reordered source acquisition')
    for member, row in zip(expected.members, supplied, strict=True):
        _need(type(row.data) is bytes, 'exact immutable bytes required')
        _need(len(row.data) == member.file_bytes, 'file length mismatch: '+member.path)
        _need(hashlib.sha256(row.data).hexdigest() == member.file_sha256, 'file SHA-256 mismatch: '+member.path)
        blob = b'blob '+str(len(row.data)).encode('ascii')+b'\0'+row.data
        _need(hashlib.sha1(blob).hexdigest() == member.git_blob, 'Git-blob mismatch: '+member.path)
    return VerifiedSourceSet(expected, supplied)


def reverify_source_set(verified, *, expected):
    _need(type(verified) is VerifiedSourceSet and type(verified.expected) is ExecutionSourceExpectations and verified.expected == expected,
          'verified set is not bound to these independent expectations')
    return verify_execution_source_bytes(expected=expected, supplied=verified.sources)
