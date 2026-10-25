"""Held-byte import correspondence for cooperative bound Python execution.

Not a general Python sandbox: interpreter, dependencies, bootstrap execution and
the independent caller are trust premises. Arbitrary hostile in-process mutation
cannot be made safe by Python hooks. All public checks fail closed on observed
hook/cache substitution; traces are session evidence, not caller capabilities.
"""
import builtins
from dataclasses import dataclass
import importlib
import importlib.abc
import importlib.util
import sys
from types import CodeType, ModuleType

from .source_binding import SourceBindingError, SourceBytes, reverify_source_set


class SourceImportError(SourceBindingError):
    pass


@dataclass(frozen=True, slots=True)
class AcceptedBootstrapModule:
    """Independently supplied execution premise, not inferred from __file__.

The trusted external bootstrap retains the actual executed code, module and
loader objects. A candidate may not designate these premises for itself.
"""
    name: str
    module: ModuleType
    loader: object
    executed_code: CodeType
    source: SourceBytes
    bootstrap_sha256: str


@dataclass(frozen=True, slots=True)
class ImportEvent:
    sequence: int
    caller: str
    target: str
    operation: str
    outcome: str


@dataclass(frozen=True, slots=True)
class BoundImportTrace:
    operation: str
    role: str
    events: tuple[ImportEvent, ...]


def _origin(member):
    return 'held-source://'+member.path


class BoundSourceLoader(importlib.abc.Loader):
    def __init__(self, session, name):
        self.session, self.name = session, name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        owner = self.session
        owner._integrity(loading=self.name)
        if owner._loaders.get(self.name) is not self or self.name not in owner._pending:
            owner._deny('<loader>', self.name, 'unbound loader dispatch')
        reverify_source_set(owner.verified, expected=owner.expected)
        member = owner._members[self.name]
        if sys.modules.get(self.name) is not module or module.__spec__.loader is not self:
            owner._deny('<loader>', self.name, 'module mapping substitution')
        code = compile(owner._bytes[member.path], _origin(member), 'exec', dont_inherit=True, optimize=0)
        self.executed_code = code
        owner._owned[self.name] = module
        owner._specs[self.name] = module.__spec__
        owner._event('<loader>', self.name, 'execute', 'started')
        try:
            exec(code, module.__dict__)
        except BaseException:
            owner._poisoned = True
            owner._event('<loader>', self.name, 'execute', 'failed')
            raise
        owner._event('<loader>', self.name, 'execute', 'completed')


class BoundSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, session):
        self.session = session

    def find_spec(self, fullname, path=None, target=None):
        owner = self.session
        if not owner._controlled(fullname):
            return None
        owner._integrity()
        if fullname not in owner._members or fullname not in owner._pending or target is not None:
            owner._deny('<finder>', fullname, 'undeclared project import/fallback/reload')
        loader = BoundSourceLoader(owner, fullname)
        owner._loaders[fullname] = loader
        spec = importlib.util.spec_from_loader(fullname, loader, origin=_origin(owner._members[fullname]),
                                              is_package=owner._members[fullname].is_package)
        if spec.submodule_search_locations is not None:
            spec.submodule_search_locations[:] = []
        return spec


class _BoundSession:
    def __init__(self, verified, expected, preloaded):
        self.expected = expected
        self.verified = reverify_source_set(verified, expected=expected)
        self._members = {m.module: m for m in expected.members if m.module is not None}
        self._bytes = {r.path: r.data for r in self.verified.sources}
        self._owned, self._specs, self._loaders = {}, {}, {}
        self._events, self._pending = [], []
        self._poisoned = False
        self._active = False
        self._entry_started = self._entry_completed = False
        self._preloaded = set()
        if type(preloaded) is not tuple:
            raise SourceImportError('immutable independent bootstrap bindings required')
        for binding in preloaded:
            if type(binding) is not AcceptedBootstrapModule or binding.name in self._preloaded:
                raise SourceImportError('duplicate/malformed bootstrap binding')
            member = self._members.get(binding.name)
            if member is None or type(binding.module) is not ModuleType or type(binding.executed_code) is not CodeType:
                raise SourceImportError('unbound bootstrap module')
            if type(binding.source) is not SourceBytes or type(binding.source.data) is not bytes or binding.source.path != member.path or binding.source.data != self._bytes[member.path]:
                raise SourceImportError('bootstrap held-source mismatch')
            if binding.bootstrap_sha256 != expected.trust_premises[0].file_sha256:
                raise SourceImportError('bootstrap premise mismatch')
            if compile(binding.source.data, _origin(member), 'exec', dont_inherit=True, optimize=0) != binding.executed_code:
                raise SourceImportError('bootstrap executed code mismatch')
            if sys.modules.get(binding.name) is not binding.module or binding.module.__spec__ is None or binding.module.__spec__.loader is not binding.loader:
                raise SourceImportError('bootstrap module/loader mapping mismatch')
            self._owned[binding.name] = binding.module
            self._specs[binding.name] = binding.module.__spec__
            self._loaders[binding.name] = binding.loader
            self._preloaded.add(binding.name)
        if expected.entry_point in self._preloaded:
            raise SourceImportError('entry must execute in this session, not a cached bootstrap')
        self._check_cache()
        self._original_import = builtins.__import__
        self._original_import_module = importlib.import_module
        self._original_profile = sys.getprofile()
        self.finder = BoundSourceFinder(self)
        self._builtin_hook = self._import
        self._module_hook = self._import_module
        self._profile_hook = self._profile

    def _controlled(self, name):
        return type(name) is str and name.split('.')[0] in self.expected.project_roots

    def _event(self, caller, target, operation, outcome):
        self._events.append(ImportEvent(len(self._events), caller, target, operation, outcome))

    def _deny(self, caller, target, reason):
        self._poisoned = True
        self._event(str(caller), str(target), reason, 'rejected')
        raise SourceImportError(reason+': '+str(target))

    def _check_cache(self, loading=None):
        for name, module in tuple(sys.modules.items()):
            if self._controlled(name) and name != loading and (name not in self._owned or self._owned[name] is not module):
                self._deny('<cache>', name, 'unbound cached module')
        for name, module in self._owned.items():
            member, spec = self._members[name], self._specs[name]
            if (sys.modules.get(name) is not module or module.__name__ != name or
                module.__spec__ is not spec or spec.name != name or spec.loader is not self._loaders[name] or
                spec.origin != _origin(member) or module.__loader__ is not self._loaders[name] or
                module.__package__ != (name if member.is_package else name.rpartition('.')[0]) or
                (member.is_package and (type(module.__path__) is not list or module.__path__ != []))):
                self._deny('<cache>', name, 'bound module mapping substitution')
            parent, _, child = name.rpartition('.')
            if parent in self._owned and not getattr(spec, '_initializing', False):
                if getattr(self._owned[parent], child, None) is not module:
                    self._deny('<cache>', name, 'parent attribute substitution')

    def _integrity(self, loading=None):
        if not self._active or self._poisoned:
            raise SourceImportError('inactive or failed import session')
        if (builtins.__import__ is not self._builtin_hook or importlib.import_module is not self._module_hook or
            sys.getprofile() is not self._profile_hook or not sys.meta_path or sys.meta_path[0] is not self.finder):
            self._deny('<hooks>', '', 'import protection replaced')
        self._check_cache(loading)

    def _caller(self, frame):
        name = frame.f_globals.get('__name__', '')
        if self._controlled(name):
            if name not in self._owned or self._owned[name].__dict__ is not frame.f_globals:
                self._deny(name, '', 'caller globals substitution')
            return name
        return '<external>'

    def _authorize(self, caller, target):
        # Unrelated trusted framework/dependency imports must remain available
        # for failure reporting after a poisoned synthetic session. They confer
        # no project-import permission and cannot produce a successful trace.
        if caller == '<external>' and not self._controlled(target):
            return
        self._integrity()
        if self._controlled(target):
            if target not in self._members:
                self._deny(caller, target, 'undeclared project import')
            if caller == '<external>' or (caller, target) not in self.expected.permitted_imports:
                self._deny(caller, target, 'undeclared project relationship')
        elif caller != '<external>' and not any(target == x or target.startswith(x+'.') for x in self.expected.external_imports):
            self._deny(caller, target, 'undeclared dependency import')

    def _parents(self, target):
        return ['.'.join(target.split('.')[:i]) for i in range(1, len(target.split('.'))+1)]

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        caller = self._caller(sys._getframe(1))
        if type(name) is not str or type(level) is not int or level < 0:
            self._deny(caller, name, 'invalid import request')
        if globals is not None and caller != '<external>' and globals is not self._owned[caller].__dict__:
            self._deny(caller, name, 'import globals substitution')
        package = (globals or {}).get('__package__', '')
        target = importlib.util.resolve_name('.'*level+name, package) if level else name
        self._authorize(caller, target)
        if caller == '<external>' and not self._controlled(target):
            return self._original_import(name, globals, locals, fromlist, level)
        targets = [target]
        if self._controlled(target):
            if fromlist is None:
                fromlist = ()  # CPython's ordinary ``import package.child``.
            if type(fromlist) not in (tuple, list) or '*' in fromlist:
                self._deny(caller, target, 'project star/invalid fromlist forbidden')
            for item in fromlist:
                if type(item) is not str:
                    self._deny(caller, target, 'invalid fromlist item')
                child = target+'.'+item
                if child in self._members:
                    self._authorize(caller, child); targets.append(child)
                elif target in self._owned:
                    value = getattr(self._owned[target], item, None)
                    if type(value) is ModuleType and self._controlled(value.__name__):
                        self._authorize(caller, value.__name__)
                        if self._owned.get(value.__name__) is not value:
                            self._deny(caller, child, 'fromlist module substitution')
        size = len(self._pending)
        for item in targets:
            if self._controlled(item): self._pending.extend(self._parents(item))
        self._event(caller, target, 'builtin_import', 'requested')
        try:
            result = self._original_import(name, globals, locals, fromlist, level)
            self._integrity()
            if self._controlled(target) and fromlist:
                # Initialization can create a module-valued attribute that did
                # not exist during the pre-load checks. Authorize the resolved
                # underlying module for the original caller before returning
                # the package to Python's IMPORT_FROM operation.
                if result is not self._owned.get(target):
                    self._deny(caller, target, 'fromlist result mapping substitution')
                for item in fromlist:
                    value = getattr(result, item, None)
                    if type(value) is ModuleType and self._controlled(value.__name__):
                        self._authorize(caller, value.__name__)
                        if self._owned.get(value.__name__) is not value:
                            self._deny(caller, target+'.'+item, 'fromlist module substitution')
            self._event(caller, target, 'builtin_import', 'completed')
            return result
        except BaseException:
            self._poisoned = True
            self._event(caller, target, 'builtin_import', 'failed')
            raise
        finally:
            del self._pending[size:]

    def _import_module(self, name, package=None):
        caller = self._caller(sys._getframe(1))
        if type(name) is not str or package is not None and type(package) is not str:
            self._deny(caller, name, 'invalid import_module request')
        target = importlib.util.resolve_name(name, package) if name.startswith('.') else name
        self._authorize(caller, target)
        if caller == '<external>' and not self._controlled(target):
            return self._original_import_module(name, package)
        return self._load(target, caller)

    def _load(self, target, caller):
        size = len(self._pending)
        if self._controlled(target): self._pending.extend(self._parents(target))
        self._event(caller, target, 'import_module', 'requested')
        try:
            result = self._original_import_module(target)
            self._integrity()
            self._event(caller, target, 'import_module', 'completed')
            return result
        except BaseException:
            self._poisoned = True
            self._event(caller, target, 'import_module', 'failed')
            raise
        finally:
            del self._pending[size:]

    def _profile(self, frame, event, arg):
        # A pre-install alias of the original importer can otherwise return a
        # cached module without invoking any finder/audit event. Reject that
        # route, while the checked wrappers remain usable for cached imports.
        if self._original_profile is not None:
            self._original_profile(frame, event, arg)
        if event == 'c_call' and arg is self._original_import and frame.f_code is not type(self)._import.__code__:
            # CPython's fromlist/parent machinery calls the original builtin
            # through this exact trusted trampoline. Permit only a target
            # already authorized by an active wrapper, never a cache exemption.
            if frame.f_code is importlib._bootstrap._call_with_frames_removed.__code__:
                arguments = frame.f_locals.get('args', ())
                previous = frame.f_back
                while previous is not None:
                    if (previous.f_code in (type(self)._import.__code__, type(self)._load.__code__)
                        and previous.f_locals.get('self') is self):
                        if (frame.f_locals.get('f') is self._original_import and type(arguments) is tuple
                            and arguments and arguments[0] in self._pending):
                            self._integrity()
                            return
                        break
                    previous = previous.f_back
            caller = self._caller(frame)
            # C profile events do not expose arguments, so an old builtin alias
            # cannot be granted an external-caller exemption: it may hit cache.
            self._deny(caller, frame.f_code.co_qualname, 'cached builtin importer bypass')
        if event == 'call' and frame.f_code is getattr(self._original_import_module, '__code__', None):
            previous = frame.f_back
            if previous is not None and previous.f_code is not type(self)._load.__code__:
                caller = self._caller(previous)
                target = frame.f_locals.get('name', '')
                if caller != '<external>' or self._controlled(target):
                    self._deny(caller, target, 'cached import_module bypass')

    def install(self):
        if self._active:
            raise SourceImportError('session already installed')
        self._check_cache()
        sys.meta_path.insert(0, self.finder)
        builtins.__import__ = self._builtin_hook
        importlib.import_module = self._module_hook
        self._active = True
        sys.setprofile(self._profile_hook)
        return self

    def import_entry(self):
        self._integrity()
        if self._entry_started:
            self._deny('<entry>', self.expected.entry_point, 'entry already invoked')
        self._entry_started = True
        result = self._load(self.expected.entry_point, '<entry>')
        self._entry_completed = True
        return result

    def trace(self):
        self._integrity()
        if not self._entry_completed:
            raise SourceImportError('entry has not completed')
        return BoundImportTrace(self.expected.operation, self.expected.role, tuple(self._events))

    def close(self):
        problems = []
        try:
            self._integrity()
        except SourceBindingError as exc:
            problems.append(str(exc))
        # Restore only our own hooks; do not overwrite another owner's changes.
        if sys.getprofile() is self._profile_hook or sys.getprofile() is None and self._poisoned:
            sys.setprofile(self._original_profile)
        else: problems.append('profile restoration unavailable')
        if builtins.__import__ is self._builtin_hook: builtins.__import__ = self._original_import
        else: problems.append('builtin hook restoration unavailable')
        if importlib.import_module is self._module_hook: importlib.import_module = self._original_import_module
        else: problems.append('module hook restoration unavailable')
        if self.finder in sys.meta_path: sys.meta_path.remove(self.finder)
        for name in sorted(set(self._owned)-self._preloaded, key=lambda n: (-n.count('.'), n)):
            module = self._owned[name]
            if sys.modules.get(name) is module:
                del sys.modules[name]
            elif name in sys.modules:
                problems.append('substituted cache preserved: '+name)
            parent, _, child = name.rpartition('.')
            if parent in self._owned and getattr(self._owned[parent], child, None) is module:
                delattr(self._owned[parent], child)
        self._active = False
        if problems:
            raise SourceImportError('; '.join(problems))


def install_bound_project_imports(verified, *, expected, preloaded=()):
    return _BoundSession(verified, expected, preloaded).install()


def validate_bound_import_trace(trace, *, session):
    if type(session) is not _BoundSession or type(trace) is not BoundImportTrace:
        raise SourceImportError('actual bound session and exact trace required')
    reverify_source_set(session.verified, expected=session.expected)
    if trace != session.trace():
        raise SourceImportError('trace differs from actual session evidence')
