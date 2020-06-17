from prolog.interpreter.term import NumberedVar

class EnumerationMemo(object):
    """A memo object to enumerate the variables in a term"""
    def __init__(self):
        self.mapping = {}
        self.numvar_in_head = {}
        self.numvar_in_body = {}
        self.numvar_twice = {}
        self.in_head = True

    def get(self, var, name = None):
        res = self.mapping.get(var, None)
        if not res:
            self.mapping[var] = res = NumberedVar(-1, name)
        else:
            self.numvar_twice[res] = None
        if self.in_head:
            self.numvar_in_head[res] = None
        else:
            self.numvar_in_body[res] = None
        return res

    def assign_numbers(self):
        # assign shared variables first
        current_index = 0
        for var in self.numvar_in_body:
            if var not in self.numvar_twice:
                continue
            if var not in self.numvar_in_head:
                continue
            var.num = current_index
            current_index += 1
        self.nshared = current_index

        # now assign variables in head only
        for var in self.numvar_in_head:
            if var not in self.numvar_twice:
                continue
            if var in self.numvar_in_body:
                continue
            var.num = current_index
            current_index += 1
        self.nhead = current_index

        # now assign variables in body only, starting at nshared again
        current_index = self.nshared
        for var in self.numvar_in_body:
            if var not in self.numvar_twice:
                continue
            if var in self.numvar_in_head:
                continue
            var.num = current_index
            current_index += 1
        self.nbody = current_index


class CopyMemo(object):
    def __init__(self):
        self.seen = None

    def get(self, key):
        if self.seen is None:
            self.seen = {}
        return self.seen.get(key, None)

    def set(self, key, val):
        if self.seen is None:
            self.seen = {}
        self.seen[key] = val

