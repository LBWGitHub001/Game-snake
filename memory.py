class Memory:
    def __init__(self):
        self.states=[]
        self.actions=[]
        self.rewards=[]
        self.logprob=[]
        self.is_terminal=[]

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprob[:]
        del self.is_terminal[:]
