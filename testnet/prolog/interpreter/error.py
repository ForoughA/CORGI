from rpython.rlib import rstring

class PrologError(Exception):
    pass

class UncatchableError(PrologError):
    def __init__(self, message):
        self.message = message

class PrologParseError(PrologError):
    def __init__(self, file_name, line_number, message):
        self.file_name = file_name
        self.line_number = line_number
        self.message = message

class TermedError(PrologError):
    def __init__(self, term, sig_context=None):
        self.term = term
        self.sig_context = sig_context

    def get_errstr(self, engine):
        from prolog.builtin import formatting
        from prolog.interpreter import term, signature
        errorsig = signature.Signature.getsignature("error", 1)

        f = formatting.TermFormatter(engine, quoted=True, max_depth=20)
        f._make_reverse_op_mapping()

        t = self.term
        if not isinstance(t, term.Callable) or not t.signature().eq(errorsig):
            return "Unhandled exception: %s" % (f.format(t), )

        errorterm = t.argument_at(0)

        if isinstance(errorterm, term.Callable):
            if errorterm.name() == "instantiation_error":
                return "arguments not sufficiently instantiated"
            elif errorterm.name() == "existence_error":
                if isinstance(errorterm, term.Callable):
                     return "Undefined %s: %s" % (
                        f.format(errorterm.argument_at(0)),
                        f.format(errorterm.argument_at(1)))
            elif errorterm.name() == "domain_error":
                if isinstance(errorterm, term.Callable):
                    return "Domain error: '%s' expected, found '%s'" % (
                        f.format(errorterm.argument_at(0)),
                        f.format(errorterm.argument_at(1)))
            elif errorterm.name() == "type_error":
                if isinstance(errorterm, term.Callable):
                    return "Type error: '%s' expected, found '%s'" % (
                        f.format(errorterm.argument_at(0)),
                        f.format(errorterm.argument_at(1)))
            elif errorterm.name() == "syntax_error":
                if isinstance(errorterm, term.Callable):
                    return "Syntax error: '%s'" % \
                    f.format(errorterm.argument_at(0))
            elif errorterm.name() == "permission_error":
                if isinstance(errorterm, term.Callable):
                    return "Permission error: '%s', '%s', '%s'" % (
                    f.format(errorterm.argument_at(0)),
                    f.format(errorterm.argument_at(1)),
                    f.format(errorterm.argument_at(2)))
            elif errorterm.name() == "representation_error":
                if isinstance(errorterm, term.Callable):
                    return "%s: Cannot represent: %s" % (
                    f.format(errorterm.argument_at(0)),
                    f.format(errorterm.argument_at(1)))
            elif errorterm.name() == "import_error":
                if isinstance(errorterm, term.Callable):
                    return "Exported procedure %s:%s is not defined" % (
                    f.format(errorterm.argument_at(0)),
                    f.format(errorterm.argument_at(1)))
            else:
                return "Internal error" # AKA, I have no clue what went wrong.

class CatchableError(TermedError): pass
class UncaughtError(TermedError):
    def __init__(self, term, sig_context=None, rule_likely_source=None, scont=None):
        TermedError.__init__(self, term, sig_context)
        self.rule = rule_likely_source
        self.traceback = _construct_traceback(scont)

    def format_traceback(self, engine):
        out = ["Traceback (most recent call last):"]
        self.traceback._format(out)
        context = ""
        if self.sig_context is not None:
            context = self.sig_context.string()
            if context == "throw/1":
                context = ""
            else:
                context += ": "
        out.append("%s%s" % (context, self.get_errstr(engine)))
        return "\n".join(out)


class TraceFrame(object):
    def __init__(self, rule, next=None):
        self.rule = rule
        self.next = next

    def __repr__(self):
        return "TraceFrame(%r, %r)" % (self.rule, self.next)

    def _format(self, out):
        rule = self.rule
        if rule.line_range is not None:
            if rule.line_range[0] + 1 ==  rule.line_range[1]:
                lines = "line %s " % (rule.line_range[0] + 1, )
            else:
                lines = "lines %s-%s " % (rule.line_range[0] + 1, rule.line_range[1])
        else:
            lines = ""
        out.append("  File \"%s\" %sin %s:%s" % (
            rule.file_name, lines,
            rule.module.name, rule.signature.string()))
        source = rule.source
        if source is not None:
            # poor man's indent
            out.append("    " + rstring.replace(source, "\n", "\n    "))
        if self.next is not None:
            self.next._format(out)

def _construct_traceback(scont):
    from prolog.interpreter.continuation import ContinuationWithRule
    if scont is None:
        return None
    next = None
    while not scont.is_done():
        if isinstance(scont, ContinuationWithRule):
            next = TraceFrame(scont.rule, next)
        scont = scont.nextcont
    return next

def wrap_error(t):
    from prolog.interpreter import term
    t = term.Callable.build("error", [t])
    return CatchableError(t)

class UnificationFailed(PrologError):
    pass

def throw_syntax_error(msg):
    from prolog.interpreter import term
    t = term.Callable.build("syntax_error", [term.Callable.build(msg)])
    raise wrap_error(t)

def throw_import_error(modulename, signature):
    from prolog.interpreter import term
    t = term.Callable.build("import_error", [term.Callable.build(modulename),
            term.Callable.build(signature.string())])
    raise wrap_error(t)

def throw_existence_error(object_type, obj):
    from prolog.interpreter import term
    t = term.Callable.build("existence_error", [term.Callable.build(object_type), obj])
    raise wrap_error(t)

def throw_instantiation_error(obj = None):
    from prolog.interpreter import term
    raise wrap_error(term.Callable.build("instantiation_error"))

def throw_representation_error(signature, msg):
    from prolog.interpreter import term
    t = term.Callable.build("representation_error",
            [term.Callable.build(signature), term.Callable.build(msg)])
    raise wrap_error(t)

def throw_type_error(valid_type, obj):
    # valid types are:
    # atom, atomic, byte, callable, character
    # evaluable, in_byte, in_character, integer, list
    # number, predicate_indicator, variable, text
    from prolog.interpreter import term
    raise wrap_error(
        term.Callable.build("type_error", [term.Callable.build(valid_type), obj]))

def throw_domain_error(valid_domain, obj):
    from prolog.interpreter import term
    # valid domains are:
    # character_code_list, close_option, flag_value, io_mode,
    # not_empty_list, not_less_than_zero, operator_priority,
    # operator_specifier, prolog_flag, read_option, source_sink,
    # stream, stream_option, stream_or_alias, stream_position,
    # stream_property, write_option
    raise wrap_error(
        term.Callable.build("domain_error", [term.Callable.build(valid_domain), obj]))

def throw_permission_error(operation, permission_type, obj):
    from prolog.interpreter import term
    # valid operations are:
    # access, create, input, modify, open, output, reposition 

    # valid permission_types are:
    # binary_stream, flag, operator, past_end_of_stream, private_procedure,
    # static_procedure, source_sink, stream, text_stream. 
    raise wrap_error(
        term.Callable.build("permission_error", [term.Callable.build(operation),
                                       term.Callable.build(permission_type),
                                       obj]))

def throw_evaluation_error(error):
    from prolog.interpreter import term
    raise wrap_error(
        term.Callable.build("evaluation_error", [term.Callable.build(error)]))

