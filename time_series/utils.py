import time


def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        try:
            ret = f(*args, **kwargs)
        except Exception as ex:
            print(ex)

        time2 = time.time()
        print('{:s} function took {:.3f} seconds'.format(f.__name__, (time2 - time1)))
        return ret
    return wrap
