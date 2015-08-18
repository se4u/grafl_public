'''
| Filename    : args.py
| Description : The module that stores the command line arguments and other miscellaneous global parameters.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 23:37:56 2015 (-0400)
| Last-Updated: Tue Aug 18 16:16:49 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 9
'''
args = None
from functools import wraps
project_dir = r'~/Dropbox/paper/grafl/'
project_pathprefix = ['',
                      project_dir+'src/experiments/']
import __builtin__
import os
def open(name, mode='rb', buffering=1024):
    for prefix in project_pathprefix:
        try:
            path = os.path.expanduser(os.path.join(prefix, name))
            print "Opening path ", str(path)
            return __builtin__.open(path,
                                    mode=mode,
                                    buffering=buffering)
        except IOError:
            pass
    raise IOError("Couldn't open %s"%name)
