'''
| Filename    : args.py
| Description : The module that stores the command line arguments and other miscellaneous global parameters.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 23:37:56 2015 (-0400)
| Last-Updated: Wed Aug 19 22:20:24 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 19
'''
args = None
from functools import wraps
project_dir = r'~/Dropbox/paper/grafl/'
project_pathprefix = ['',
                      project_dir,
                      project_dir + 'res/experiments/',
                      project_dir + 'res/']
import __builtin__
builtin_open = __builtin__.open
import os


def open(name, mode='rb', buffering=1024):
    for prefix in project_pathprefix:
        try:
            path = os.path.expanduser(os.path.join(prefix, name))
            f = builtin_open(path, mode=mode, buffering=buffering)
            if (path.startswith(project_dir)
                    or 'pylearn2' in path
                    or 'Lasagne' in path
                    # or 'theano' in path
                    or path[0] != r'/'):
                print "Opened file ", str(path)
            return f
        except IOError:
            pass
    raise IOError("Couldn't open %s" % name)
__builtin__.open = open
