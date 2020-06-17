import py
import os
from prolog.builtin.sourcehelper import get_source
from prolog.interpreter.test.tool import collect_all, assert_false, assert_true
from prolog.interpreter.test.tool import prolog_raises, create_file, delete_file
from prolog.interpreter.error import CatchableError

def test_get_source():
    content = "some important content"
    name = "__testfile__"
    try:
        create_file(name, content)
        source, file_name = get_source(name)
    finally:
        delete_file(name)
    assert source == content
    assert file_name == os.path.abspath(name)

def test_source_does_not_exist():
    py.test.raises(CatchableError, "get_source('this_file_does_not_exist')")

def test_file_ending():
    content = "some content"
    filename = "__testfile__.pl"
    searchname = filename[:len(filename) - 3]
    try:
        create_file(filename, content)
        source, file_name = get_source(searchname)
    finally:
        delete_file(filename)
    assert source == content
    assert file_name == os.path.abspath(filename)



