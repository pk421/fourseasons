from cProfile import Profile
import pstats

def profile(sort='cumulative', lines=None, strip_dirs=True):
    """
    This allows for a decorator to be inserted into functions that will only profile that function and any sub function
    calls made from there. As a result, you *should not* explicitly run the "--profile" option from the make file.
    Simply add the @profile decorator after importing this file and then run test_run.py as usual. This will generate
    an AAAprofile.stats file which can then be pushed into a png file using gprof2dot.
    """

    print "\n***********profiling***********"
    """A decorator which profiles a callable.
    Example usage:

    >>> @profile
    ... def factorial(n):
    ...     n = abs(int(n))
    ...     if n < 1:
    ...             n = 1
    ...     x = 1
    ...     for i in range(1, n+1):
    ...             x = i * x
    ...     return x
    ...
    >>> factorial(5)
    Thu Jul 15 20:58:21 2010    /tmp/tmpIDejr5

             4 function calls in 0.000 CPU seconds

       Ordered by: internal time, call count

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    0.000    0.000 profiler.py:120(factorial)
            1    0.000    0.000    0.000    0.000 {range}
            1    0.000    0.000    0.000    0.000 {abs}
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    120
    >>>
    """
    def outer(fun):
        print "***fun: ", fun
        def inner(*args, **kwargs):
            
            # fun_name = str(fun).split('built-in function ')[1]
            # fun_name = fun_name.split('>')[0]
            # output_file = '.'.join(['AAA', fun_name, 'stats'])
            
            file = open('AAAprofile.stats', 'wb')
            prof = Profile()
            try:
                ret = prof.runcall(fun, *args, **kwargs)
            except:
                file.close()
                raise

            prof.dump_stats(file.name)
            stats = pstats.Stats(file.name)
            if strip_dirs:
                stats.strip_dirs()
            if isinstance(sort, (tuple, list)):
                stats.sort_stats(*sort)
            else:
                stats.sort_stats(sort)
            #stats.print_stats(lines)

            file.close()
            return ret
        return inner

    # in case this is defined as "@profile" instead of "@profile()"
    if hasattr(sort, '__call__'):
        fun = sort
        sort = 'cumulative'
        outer = outer(fun)

    return outer