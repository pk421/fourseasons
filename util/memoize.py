import collections
import functools

class memoize(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args, **kwargs):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args, **kwargs)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args, **kwargs)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


class MemoizeMutable:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args, **kwds):
        import cPickle
        str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
        if not self.memo.has_key(str): 
            # print "miss"  # DEBUG INFO
            self.memo[str] = self.fn(*args, **kwds)
        # else:
            # print "hit"  # DEBUG INFO

        return self.memo[str]