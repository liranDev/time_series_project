import time


def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} seconds'.format(f.__name__, (time2 - time1)))
        return ret
    return wrap
