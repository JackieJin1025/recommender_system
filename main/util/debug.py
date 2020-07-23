import logging
import time

class Timer(object):
    get_time = time.perf_counter

    def __init__(self):
        self.t0 = Timer.get_time()

    def mark(self, reset=True):
        t1 = Timer.get_time()
        e1 = t1 - self.t0
        if reset:
            self.t0 = t1

        return e1

    def restart(self):
        e = self.mark()
        return e


class LogUtil:
    __LOG__ = dict()

    @classmethod
    def getLogger(cls, name):
        if name in LogUtil.__LOG__:
            log = LogUtil.__LOG__.get(name)
            return log
        # configure root logger
        logging.basicConfig(level=logging.DEBUG, handlers=[])

        # config logger
        logger = logging.getLogger(name)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        LogUtil.__LOG__[name] = logger
        return logger




if __name__ == '__main__':
    t = Timer()
    t0 = time.time()
    time.sleep(1.0)
    t1 = time.time()
    e0 = t.restart()
    e1 = t1 - t0
    print(e0, e1)