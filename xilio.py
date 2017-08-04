from os import chmod
import sys

try:
    import sh

    def copy(s):
        if sys.platform.startswith('linux'): # for Ubuntu
            sh.xclip("-selection", "clipboard", _in=s, _bg=True)
            sh.xclip("-selection", "primary", _in=s, _bg=True)
        elif sys.platform == 'darwin':
            sh.pbcopy(_in=s, _bg=True)

    def paste(s):
        if sys.platform.startswith('linux'): # for Ubuntu
            return sh.xclip("-selection", "clipboard", "-o", _in=s)
        elif sys.platform == 'darwin':
            return sh.pbpaste()

except ImportError:
    pass



def load(path, mode='rb'):
    with open (path, mode) as file:
        from pickle import load
        return load(file)

def dump(path, obj, mode='wb'):
    with open (path, mode) as file:
        from pickle import dump
        return dump(obj, file)

def read(path, mode='r'):
    with open (path, mode) as file:
        return file.read()

def readline(path, mode='r'):
    with open (path, mode) as file:
        yield from file.readline()

def write(path, content, mode='w', executable=False):
    with open (path, mode) as file:
        file.write(content)
    if executable:
        chmod(path, 0o755)

def append(path, content, mode='a'):
    with open (path, mode) as file:
        file.write(content)