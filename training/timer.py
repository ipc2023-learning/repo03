import contextlib
import os
import sys
import time
import datetime

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.start_clock = self._clock()

    def _clock(self):
        times = os.times()
        return times[0] + times[1]

    def __str__(self):
        return "[%.3fs CPU, %.3fs wall-clock]" % (
            self._clock() - self.start_clock,
            time.time() - self.start_time)

    def wctime(self):
        return time.time() - self.start_time

    def time(self):
        return self._clock() - self.start_clock


class CountdownWCTimer:
    def __init__(self, time_limit):
        self.timer = Timer()
        self.time_limit = time_limit

    def expired(self):
        return self.timer.wctime() > self.time_limit

    def remaining_seconds (self):
        return self.time_limit - self.timer.wctime()

    def remaining_minutes (self):
        return self.remaining_seconds()/60

    def __str__(self):
        return str(datetime.timedelta(seconds=self.remaining_seconds()))


@contextlib.contextmanager
def timing(text, block=False):
    timer = Timer()
    if block:
        print("%s..." % text)
    else:
        print("%s..." % text, end=' ')
    sys.stdout.flush()
    yield
    if block:
        print("%s: %s" % (text, timer))
    else:
        print(timer)
    sys.stdout.flush()
