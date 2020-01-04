import numpy as np
from collections import deque
import random

class replay_buffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, sample):
        self.buffer.append(sample)

    def sample(self, no_samples):
        return random.sample(self.buffer, no_samples)

    def num_samples(self):
        return len(self.buffer)
    
class frame_stack:
    def __init__(self, max_size, frame_shape):
        self.stacked_frames = deque([np.zeros(frame_shape, dtype=np.int) for i in range(max_size)], maxlen=max_size)
        
    def query(self, state, done):
        if(done):
            for i in range(stack_size-1):
                stacked_frames.append(state)
                
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=2) 
        return stacked_state
       