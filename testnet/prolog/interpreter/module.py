import py
from rpython.rlib import jit
from prolog.interpreter.signature import Signature
from prolog.interpreter import error, term
from prolog.interpreter.term import Callable, Atom
from prolog.interpreter.function import Function, _make_toplevel_rule
from prolog.interpreter.helper import unwrap_predicate_indicator

class VersionTag(object):
    pass

class ModuleWrapper(object):
    _immutable_fields_ = ["version?"]

    def __init__(self, engine):
        self.engine = engine
        self.user_module = Module("user")
        self.modules = {"user": self.user_module} # all known modules
        self.seen_modules = {}
        self.current_module = self.user_module
        self.libs = []
        self.system = None
        self.version = VersionTag()

    def init_system_module(self):
        from prolog.builtin.sourcehelper import get_source
        source, file_name = get_source("system.pl")
        self.engine.runstring(source, file_name)
        self.system = self.modules["system"]
        self.current_module = self.user_module

    def get_module(self, name, errorterm):
        module = self._get_module(name, self.version)
        if module is not None:
            return module
        assert isinstance(errorterm, Callable)
        error.throw_existence_error("procedure",
            errorterm.get_prolog_signature())

    def get_or_make_module(self, name):
        module = self._get_module(name, self.version)
        if module is not None:
            return module


    @jit.elidable
    def _get_module(self, name, version):
        return self.modules.get(name, None)

    def add_module(self, name, exports = []):
        mod = Module(name)
        for export in exports:
            mod.exports.append(Signature.getsignature(
                    *unwrap_predicate_indicator(export)))
        self.current_module = mod
        self.modules[name] = mod
        self.version = VersionTag()


class Module(object):
    _immutable_fields_ = ["name", "nameatom", "_toplevel_rule"]
    def __init__(self, name):
        self.name = name
        self.nameatom = Atom(name)
        self.functions = {}
        self.exports = []
        self._toplevel_rule = _make_toplevel_rule(self)

    def add_meta_predicate(self, signature, arglist):
        func = self.lookup(signature)
        func.meta_args = arglist

    @jit.elidable_promote("0")
    def lookup(self, signature):
        try:
            function = self.functions[signature]
        except KeyError:
            function = Function()
            self.functions[signature] = function
        return function
    

    def use_module(self, module, imports=None):
        if imports is None:
            importlist = module.exports
        else:
            importlist = []
            for pred in imports:
                if pred in module.exports:
                    importlist.append(pred)
        for sig in importlist:
            try:
                self.functions[sig] = module.functions[sig]
            except KeyError:
                pass

    def get_all_rules(self):
        def get_rules(chain):
            r = []
            while chain:
                r.append(chain.source)
                chain = chain.next
            return r
        allrules = []
        for f in self.functions:
            rules = get_rules(self.lookup(f).rulechain)
            allrules += rules

        return allrules
            
              

    def __repr__(self):
        return "Module('%s')" % self.name
        
