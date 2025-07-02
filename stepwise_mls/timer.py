import time


class Timer:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.verbose:
            self.tell_time()

    def tell_time(self):
        print(f'Elapsed time: {self.elapsed_time: .4f} seconds')