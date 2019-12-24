import collections
import functools
import time

class memoize(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.max_cached_items = 10
      self.item_count = 0
      self.func = func
      self.cache = {}
   def __call__(self, *args, **kwargs):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args, **kwargs)
      if args in self.cache:
         # cache hit
         current_time = time.time()
         # update the time
         self.cache[args][1] = current_time

         return self.cache[args][0]
      elif self.item_count < self.max_cached_items:
         # cache miss and add
         value = self.func(*args, **kwargs)
         current_time = time.time()
         self.cache[args] = [value, current_time]

         self.item_count += 1

         return value
      else:
          # cache miss - add and evict old
          value = self.func(*args, **kwargs)
          current_time = time.time()
          self.cache[args] = [value, current_time]

          # This is really inefficient but super easy...not so terrible though if cache hits are far more common than
          # cache misses, which would be the case here
          oldest_timestamp = sorted([v[1] for v in self.cache.values()])[0]
          self.cache = { k : v for k, v in self.cache.iteritems() if v[1] != oldest_timestamp }

          return value

   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


# class MemoizeMutable:
#     def __init__(self, fn):
#         self.fn = fn
#         self.memo = {}
#     def __call__(self, *args, **kwds):
#         import cPickle
#         str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
#         if not self.memo.has_key(str): 
#             # print "miss"  # DEBUG INFO
#             self.memo[str] = self.fn(*args, **kwds)
#         # else:
#             # print "hit"  # DEBUG INFO

#         return self.memo[str]



# class MemoizeMutable:
#     """
#       This will only cache a few items, see the elif statement.
#     """
#     def __init__(self, fn):
#         self.fn = fn
#         self.memo = {}
#         self.item_count = 0
#     def __call__(self, *args, **kwds):
#         import cPickle
#         str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
#         if self.memo.has_key(str):
#             return self.memo[str]
        
#         elif self.item_count <= 1: 
#             # print "miss"  # DEBUG INFO
#             self.memo[str] = self.fn(*args, **kwds)
#             self.item_count += 1
#         else:
#           return self.fn(*args, **kwds)
#         # else:
#             # print "hit"  # DEBUG INFO

#         return self.memo[str]




class MemoizeMutable:
    """
      This will only cache a few items, see the elif statement.
      New algo is such that we *know* for most of these calls they will alternate between a fixed stock and a variable
      stock. As such, we expect to see the fixed stock every other call. Thus, we only cache something if we have
      seen it precisely two calls ago.
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
        self.item_count = 0
        self.max_cached_items = 1
        self.one_fn_ago = fn
        self.two_fns_ago = fn
        # self.args_one_call_ago = []
        # self.kwargs_one_call_ago = {}
        # self.args_two_calls_ago = []
        # self.kwargs_two_calls_ago = {}
    def __call__(self, *args, **kwds):
        import cPickle
        str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
        if self.memo.has_key(str):
            # it was previously cached, and there is no need to update it
            # print "reading cache, has key already", args, kwds
            self.two_fns_ago = self.one_fn_ago
            self.one_fn_ago = str
            return self.memo[str]
        
        elif self.item_count < self.max_cached_items:
            # this is for the very first run to set up the caching when the history is not established
            # notice everything gets set
            print "first run memoization", args, kwds
            self.memo.clear()
            self.memo[str] = self.fn(*args, **kwds)
            self.two_fns_ago = str
            self.one_fn_ago = str
            self.item_count += 1
            return self.memo[str]

        elif str == self.two_fns_ago:
            # cache the item, basically, this is the second time we are seeing it
            # print "caching now", args, kwds
            self.memo.clear()
            self.memo[str] = self.fn(*args, **kwds)
            self.two_fns_ago = self.one_fn_ago
            self.one_fn_ago = str
            self.item_count += 1
            return self.memo[str]

        else:
          # the item was not cached and will not be cached now
          "not caching now memoization", args, kwds
          self.two_fns_ago = self.one_fn_ago
          self.one_fn_ago = str
          return self.fn(*args, **kwds)
        # else:
            # print "hit"  # DEBUG INFO

        return self.memo[str]







# class MemoizeMutable:
#     """
#       This will only cache a few items, see the elif statement.
#       New algo is such that we *know* for most of these calls they will alternate between a fixed stock and a variable
#       stock. As such, we expect to see the fixed stock every other call. Thus, we only cache something if we have
#       seen it precisely two calls ago.
#     """
#     def __init__(self, fn):
#         self.fn = fn
#         self.memo = {}
#         self.item_count = 0
#         self.max_cached_items = 1
#         self.one_fn_ago = fn
#         self.two_fns_ago = fn
#         # self.args_one_call_ago = []
#         # self.kwargs_one_call_ago = {}
#         # self.args_two_calls_ago = []
#         # self.kwargs_two_calls_ago = {}
#     def __call__(self, *args, **kwds):
#         import cPickle
#         str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
#         if self.memo.has_key(str):
#             # it was previously cached, and there is no need to update it
#             # print "reading cache, has key already", args, kwds
#             self.two_fns_ago = self.one_fn_ago
#             self.one_fn_ago = self.fn(*args, **kwds)
#             return self.memo[str]
        
#         elif self.item_count < self.max_cached_items:
#             # this is for the very first run to set up the caching when the history is not established
#             # notice everything gets set
#             print "first run", args, kwds
#             self.memo[str] = self.fn(*args, **kwds)
#             self.two_fns_ago = self.fn(*args, **kwds)
#             self.one_fn_ago = self.fn(*args, **kwds)
#             self.item_count += 1
#             return self.memo[str]

#         elif self.fn(*args, **kwds) == self.two_fns_ago:
#             # cache the item, basically, this is the second time we are seeing it
#             # print "miss"  # DEBUG INFO
#             print "caching now", args, kwds
#             self.memo[str] = self.fn(*args, **kwds)
#             self.two_fns_ago = self.one_fn_ago
#             self.one_fn_ago = self.fn(*args, **kwds)
#             self.item_count += 1
#             return self.memo[str]

#         else:
#           # the item was not cached and will not be cached now
#           "not caching now", args, kwds
#           self.two_fns_ago = self.one_fn_ago
#           self.one_fn_ago = self.fn(*args, **kwds)
#           return self.fn(*args, **kwds)
#         # else:
#             # print "hit"  # DEBUG INFO

#         return self.memo[str]


# Memoization with settable size of cache
# Taken From: http://scottlobdell.me/2015/04/decorators-arguments-python/
from collections import deque


class Memoized(object):

    def __init__(self, cache_size=100):
        self.cache_size = cache_size
        self.call_args_queue = deque()
        self.call_args_to_result = {}

    def __call__(self, fn, *args, **kwargs):

        def new_func(*args, **kwargs):
            memoization_key = self._convert_call_arguments_to_hash(args, kwargs)
            if memoization_key not in self.call_args_to_result:
                result = fn(*args, **kwargs)
                self._update_cache_key_with_value(memoization_key, result)
                self._evict_cache_if_necessary()
            return self.call_args_to_result[memoization_key]

        return new_func

    def _update_cache_key_with_value(self, key, value):
        self.call_args_to_result[key] = value
        self.call_args_queue.append(key)

    def _evict_cache_if_necessary(self):
        if len(self.call_args_queue) > self.cache_size:
            oldest_key = self.call_args_queue.popleft()
            del self.call_args_to_result[oldest_key]

    @staticmethod
    def _convert_call_arguments_to_hash(args, kwargs):
        return hash(str(args) + str(kwargs))


@Memoized(cache_size=5)
def get_not_so_random_number_with_max(max_value):
    import random
    return random.random() * max_value