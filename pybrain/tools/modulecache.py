'''
Created on Mar 25, 2013

@author: kaeufl
'''
import hashlib

class ModuleCache(object):
    """
    Wraps any module object and caches its activate output.
    """
    def __init__(self, module):
        self._module = module
        self._cache = {}
        
    def __getattr__(self, name):
        attr = getattr(self._module, name)
        if name == 'activate':
            def activate(inpt):
                key = hashlib.sha1(inpt).hexdigest()
                if not key in self._cache.keys():
                    self._cache[key] = attr(inpt)
                return self._cache[key]
            return activate
        return attr