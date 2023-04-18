import collections
import itertools
import random
import typing
from scipy import stats   
import math

class KSWIN_std:

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = None,
        window: typing.Iterable = None,
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._drift_detected= False

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    def _reset(self):
        #super()._reset()
        self.p_value = 0
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)
        self._drift_detected= False

    def update(self, x):

        if self._drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:
            rnd_window = [
                self.window[r]
                for r in self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
            ]
            most_recent = list(
                itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
            )

            st, self.p_value = stats.ks_2samp(rnd_window, most_recent,method='exact')

            if self.p_value <= self.alpha and st > 0.1:
                self._drift_detected = True
                self.window = collections.deque(most_recent, maxlen=self.window_size)
            else:
                self._drift_detected = False
        else:  # Not enough samples in the sliding window for a valid test
            self._drift_detected = False

        return self
    


class KSWIN_std_D:

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = None,
        window: typing.Iterable = None,
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._drift_detected= False
        self.threshold = (-math.log(alpha/2)/stat_size)**0.5 # "/21" is correct
        self.st = 0

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    def _reset(self):
        #super()._reset()
        self.p_value = 0
        self.st = 0
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)
        self._drift_detected= False

    def update(self, x):

        if self._drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:
            rnd_window = [
                self.window[r]
                for r in self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
            ]
            most_recent = list(
                itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
            )

            self.st, self.p_value = stats.ks_2samp(rnd_window, most_recent,method='exact')

            #if self.p_value <= self.alpha and st > 0.1:
            if self.st >= self.threshold and self.st > 0.1:
                self._drift_detected = True
                self.window = collections.deque(most_recent, maxlen=self.window_size)
            else:
                self._drift_detected = False
        else:  # Not enough samples in the sliding window for a valid test
            self._drift_detected = False

        return self


class KSWIN_soft_D:

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = None,
        th_r: float=0.5,
        window: typing.Iterable = None,
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._drift_detected= False
        self.threshold = (-math.log(alpha/2)/stat_size)**0.5 # "/21" is correct
        self.st = 0
        self.th_r=th_r
        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    def _reset(self):
        #super()._reset()
        self.p_value = 0
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)
        self._drift_start= False
        self.drift_success_count=0
        self.drift_win_count=0
        self.drift_detected = False
        self.st = 0
        
    def update(self, x):
        
        if self._drift_detected:
            self._reset()
            
        self.n += 1

        self.window.append(x)
        
        
        if len(self.window) >= self.window_size:
            
            if self._drift_start == False:
                rnd_window = [
                    self.window[r]
                    for r in self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
                ]
                most_recent = list(
                    itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
                )
    
                self.st, self.p_value = stats.ks_2samp(rnd_window, most_recent,method='exact')
    
                #if self.p_value <= self.alpha and st > 0.1:
                if self.st >= self.threshold and self.st > 0.1:
                    self._drift_start = True
                    self.drift_success_count=1
                    self.drift_win_count=1
                
            else:
                rnd_window = [
                    self.window[r]
                    for r in self._rng.sample(range(self.window_size - self.stat_size-self.drift_win_count), self.stat_size)
                ]
                most_recent = list(
                    itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
                )
                
                self.st, self.p_value = stats.ks_2samp(rnd_window, most_recent,method='exact')
    
                #if self.p_value <= self.alpha and st > 0.1:
                if self.st >= self.threshold and self.st > 0.1:
                    self.drift_success_count+=1
                
                self.drift_win_count+=1
                
                if self.drift_win_count>=self.stat_size:
                    ratio=self.drift_success_count/self.drift_win_count
                    if ratio>self.th_r:
                        self._drift_detected = True
                        #self.window = collections.deque(most_recent, maxlen=self.window_size)
                    else:
                        self._drift_detected = False
                        self.drift_success_count=0
                        self.drift_win_count=0
                              
                    self._drift_start == False
                    
        else:  # Not enough samples in the sliding window for a valid test
            self._drift_detected = False

        return self


    
class KSWIN_soft:

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = None,
        window: typing.Iterable = None,
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._drift_detected= False

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    def _reset(self):
        #super()._reset()
        self.p_value = 0
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)
        self._drift_start= False
        self.drift_success_count=0
        self.drift_win_count=0
        self.drift_detected = False

    def update(self, x):
        
        if self._drift_detected:
            self._reset()
            
        self.n += 1

        self.window.append(x)
        
        
        if len(self.window) >= self.window_size:
            
            if self._drift_start == False:
                rnd_window = [
                    self.window[r]
                    for r in self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
                ]
                most_recent = list(
                    itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
                )
    
                st, self.p_value = stats.ks_2samp(rnd_window, most_recent,method='exact')
    
                if self.p_value <= self.alpha and st > 0.1:
                    self._drift_start = True
                    self.drift_success_count=1
                    self.drift_win_count=1
                
            else:
                rnd_window = [
                    self.window[r]
                    for r in self._rng.sample(range(self.window_size - self.stat_size-self.drift_win_count), self.stat_size)
                ]
                most_recent = list(
                    itertools.islice(self.window, self.window_size - self.stat_size, self.window_size)
                )
                
                st, self.p_value = stats.ks_2samp(rnd_window, most_recent,method='exact')
    
                if self.p_value <= self.alpha and st > 0.1:
                    self.drift_success_count+=1
                
                self.drift_win_count+=1
                
                if self.drift_win_count>=self.stat_size:
                    ratio=self.drift_success_count/self.drift_win_count
                    if ratio>0.5:
                        self._drift_detected = True
                        #self.window = collections.deque(most_recent, maxlen=self.window_size)
                    else:
                        self._drift_detected = False
                        self.drift_success_count=0
                        self.drift_win_count=0
                              
                    self._drift_start == False
                    
        else:  # Not enough samples in the sliding window for a valid test
            self._drift_detected = False

        return self
