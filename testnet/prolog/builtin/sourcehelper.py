import os
import sys
from prolog.interpreter.error import throw_existence_error
from prolog.interpreter.term import Callable

path = os.path.dirname(__file__)
path = os.path.join(path, "..", "prolog_modules")

def get_source(filename):
    try:
        assert isinstance(filename, str)
        fd, actual_filename = get_filehandle(filename, True)
    except OSError:
        throw_existence_error("source_sink", Callable.build(filename))
        assert 0, "unreachable" # make the flow space happy
    try:
        content = []
        while 1:
            s = os.read(fd, 4096)
            if not s:
                break
            content.append(s)
        file_content = "".join(content)
    finally:
        os.close(fd)
    return file_content, actual_filename

def get_filehandle(filename, stdlib=False):
    filename_with_pl =  filename + '.pl'
    candidates = [
        filename,
        filename_with_pl,
        os.path.join(path, filename),
        os.path.join(path, filename_with_pl)]
    e = None
    for cand in candidates:
        try:
            return os.open(cand, os.O_RDONLY, 0777), os.path.abspath(cand)
        except OSError, e:
            pass
    assert e is not None
    raise e
