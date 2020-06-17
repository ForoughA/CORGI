from rpython.rlib  import jit, debug, objectmodel

def inline_small_list(sizemax=11, sizemin=0, immutable=False, attrname="list", factoryname="make", unbox_num=False):
    """
    This function is helpful if you have a class with a field storing a
    list and the list is often very small. Calling this function will inline
    the list into instances for the small sizes. This works by adding the
    following methods to the class:

    _get_list(self, i): return ith element of the list

    _set_list(self, i, val): set ith element of the list

    _get_full_list(self): returns a copy of the full list

    @staticmethod
    make(listcontent, *args): makes a new instance with the list's content set to listcontent
    """
    def wrapper(cls):
        from rpython.rlib.unroll import unrolling_iterable
        classes = []
        def make_methods(size):
            attrs = ["_%s_%s" % (attrname, i) for i in range(size)]
            unrolling_enumerate_attrs = unrolling_iterable(enumerate(attrs))
            def _get_size_list(self):
                return size
            def _get_list(self, i):
                for j, attr in unrolling_enumerate_attrs:
                    if j == i:
                        return getattr(self, attr)
                raise IndexError
            def _get_full_list(self):
                res = [None] * size
                for i, attr in unrolling_enumerate_attrs:
                    res[i] = getattr(self, attr)
                return res
            def _set_list(self, i, val):
                for j, attr in unrolling_enumerate_attrs:
                    if j == i:
                        setattr(self, attr, val)
                        return
                raise IndexError
            def _init(self, elems, *args):
                assert len(elems) == size
                for i, attr in unrolling_enumerate_attrs:
                    setattr(self, attr, elems[i])
                cls.__init__(self, *args)
            meths = {"_get_list": _get_list, "_get_size_list": _get_size_list, "_get_full_list": _get_full_list, "_set_list": _set_list, "__init__" : _init}
            if immutable:
                meths["_immutable_fields_"] = attrs
            return meths
        classes = [type(cls)("%sSize%s" % (cls.__name__, size), (cls, ), make_methods(size)) for size in range(sizemin, sizemax)]
        def _get_arbitrary(self, i):
            return getattr(self, attrname)[i]
        def _get_size_list_arbitrary(self):
            return len(getattr(self, attrname))
        def _get_list_arbitrary(self):
            return getattr(self, attrname)
        def _set_arbitrary(self, i, val):
            getattr(self, attrname)[i] = val
        def _init(self, elems, *args):
            debug.make_sure_not_resized(elems)
            setattr(self, attrname, elems)
            cls.__init__(self, *args)
        meths = {"_get_list": _get_arbitrary, "_get_size_list": _get_size_list_arbitrary, "_get_full_list": _get_list_arbitrary, "_set_list": _set_arbitrary, "__init__": _init}
        if immutable:
            meths["_immutable_fields_"] = ["%s[*]" % (attrname, )]
        cls_arbitrary = type(cls)("%sArbitrary" % cls.__name__, (cls, ), meths)

        def make(elems, *args):
            if classes:
                if (elems is None or len(elems) == 0):
                    return make0(*args)
            else:
                if elems is None:
                    elems = []
            if sizemin <= len(elems) < sizemax:
                cls = classes[len(elems) - sizemin]
            else:
                cls = cls_arbitrary
            return cls(elems, *args)

        # XXX those could be done more nicely
        def make0(*args):
            if not classes: # no type specialization
                return make([], *args)
            result = objectmodel.instantiate(classes[0])
            cls.__init__(result, *args)
            return result
        def make1(elem, *args):
            if not classes: # no type specialization
                return make([elem], *args)
            result = objectmodel.instantiate(classes[1])
            result._set_list(0, elem)
            cls.__init__(result, *args)
            return result
        def make2(elem1, elem2, *args):
            if not classes: # no type specialization
                return make([elem1, elem2], *args)
            result = objectmodel.instantiate(classes[2])
            result._set_list(0, elem1)
            result._set_list(1, elem2)
            cls.__init__(result, *args)
            return result

        def make_n(size, *args):
            if sizemin <= size < sizemax:
                subcls = classes[size - sizemin]
            else:
                subcls = cls_arbitrary
            result = objectmodel.instantiate(subcls)
            if subcls is cls_arbitrary:
                assert isinstance(result, subcls)
                setattr(result, attrname, [None] * size)
            cls.__init__(result, *args)
            return result

        if unbox_num:
            make, make1, make2 = _add_num_classes(cls, make, make0, make1, make2)
        setattr(cls, factoryname, staticmethod(make))
        setattr(cls, factoryname + "0", staticmethod(make0))
        setattr(cls, factoryname + "1", staticmethod(make1))
        setattr(cls, factoryname + "2", staticmethod(make2))
        setattr(cls, factoryname + "_n", staticmethod(make_n))
        return cls
    return wrapper

def _add_num_classes(cls, orig_make, orig_make0, orig_make1, orig_make2):
    # XXX quite brute force
    def make(vals, *args):
        from prolog.interpreter.term import Number
        if vals is None or len(vals) == 0:
            return orig_make0(*args)
        if len(vals) == 1:
            return make1(vals[0], *args)
        if len(vals) == 2:
            return make2(vals[0], vals[1], *args)
        return orig_make(vals, *args)
    def make1(w_a, *args):
        from prolog.interpreter.term import Number, Float
        if isinstance(w_a, Number):
            return Size1Fixed(w_a.num, *args)
        if isinstance(w_a, Float):
            return Size1Flo(w_a.floatval, *args)
        return orig_make1(w_a, *args)
    def make2(w_a, w_b, *args):
        from prolog.interpreter.term import Number
        if isinstance(w_a, Number):
            if isinstance(w_b, Number):
                return Size2Fixed11(w_a.num, w_b.num, *args)
            else:
                return Size2Fixed10(w_a.num, w_b, *args)
        elif isinstance(w_b, Number):
            return Size2Fixed01(w_a, w_b.num, *args)
        return orig_make2(w_a, w_b, *args)

    class Size1Fixed(cls):
        def __init__(self, vals_fixed_0, *args):
            self.vals_fixed_0 = vals_fixed_0
            cls.__init__(self, *args)

        def _get_size_list(self):
            return 1

        def _get_full_list(self):
            return [self._get_list(0)]

        def _get_list(self, i):
            from prolog.interpreter.term import Number
            assert i == 0
            return Number(self.vals_fixed_0)

        def _set_list(self, i, val):
            raise NotImplementedError()
    Size1Fixed.__name__ = cls.__name__ + Size1Fixed.__name__

    class Size1Flo(cls):
        def __init__(self, vals_flo_0, *args):
            self.vals_flo_0 = vals_flo_0
            cls.__init__(self, *args)

        def _get_size_list(self):
            return 1

        def _get_full_list(self):
            return [self._get_list(0)]

        def _get_list(self, i):
            from prolog.interpreter.term import Float
            assert i == 0
            return Float(self.vals_flo_0)

        def _set_list(self, i, val):
            raise NotImplementedError()
    Size1Flo.__name__ = cls.__name__ + Size1Flo.__name__

    class Size2Fixed10(cls):
        def __init__(self, vals_fixed_0, w_val1, *args):
            self.vals_fixed_0 = vals_fixed_0
            self.w_val1 = w_val1
            cls.__init__(self, *args)

        def _get_size_list(self):
            return 2

        def _get_full_list(self):
            return [self._get_list(0), self._get_list(1)]

        def _get_list(self, i):
            from prolog.interpreter.term import Number
            if i == 0:
                return Number(self.vals_fixed_0)
            else:
                assert i == 1
                return self.w_val1

        def _set_list(self, i, val):
            raise NotImplementedError()
    Size2Fixed10.__name__ = cls.__name__ + Size2Fixed10.__name__


    class Size2Fixed01(cls):
        def __init__(self, w_val0, vals_fixed_1, *args):
            self.w_val0 = w_val0
            self.vals_fixed_1 = vals_fixed_1
            cls.__init__(self, *args)

        def _get_size_list(self):
            return 2

        def _get_full_list(self):
            return [self._get_list(0), self._get_list(1)]

        def _get_list(self, i):
            from prolog.interpreter.term import Number
            if i == 0:
                return self.w_val0
            else:
                assert i == 1
                return Number(self.vals_fixed_1)

        def _set_list(self, i, val):
            raise NotImplementedError()
    Size2Fixed01.__name__ = cls.__name__ + Size2Fixed01.__name__

    class Size2Fixed11(cls):
        def __init__(self, vals_fixed_0, vals_fixed_1, *args):
            self.vals_fixed_0 = vals_fixed_0
            self.vals_fixed_1 = vals_fixed_1
            cls.__init__(self, *args)

        def _get_size_list(self):
            return 2

        def _get_full_list(self):
            return [self._get_list(0), self._get_list(1)]

        def _get_list(self, i):
            from prolog.interpreter.term import Number
            if i == 0:
                return Number(self.vals_fixed_0)
            else:
                assert i == 1
                return Number(self.vals_fixed_1)

        def _set_list(self, i, val):
            raise NotImplementedError()
    Size2Fixed11.__name__ = cls.__name__ + Size2Fixed11.__name__

    return make, make1, make2
