import errno
import shutil

def copyanything(src, dst,symlinks=True):
    try:
        shutil.copytree(src, dst,symlinks)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
                        shutil.copy(src, dst)
        else: raise
