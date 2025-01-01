class Logger:
    def __init__(self):
        self.infoDataGenrated = list()
        self.config = None
        self.epoch = 0
        self.eposide = 0

    def setConfig(self, config):
        self.config = config

    def printInfo(self):
        # 输出线程情况
        epochInfo = '{'
        once = int(20 / self.config['epoch'])
        for _ in range(self.epoch):
            for _ in range(once):
                epochInfo += '|'
        for _ in range(self.config['epoch'] - self.epoch):
            for _ in range(once):
                epochInfo += ' '
        epochInfo += '}'
        eposideInfo = 'Eposide[' + str(self.eposide) + '/' + str(self.config['eposide']*self.config['num_games']) + ']'
        threadInfo = 'ThreadInfo:'
        for info in self.infoDataGenrated:  # int info
            threadInfo = threadInfo + '\t[' + str(info) + '/' + str(self.config['eposide']) + ']'
        info = eposideInfo + ' ' + threadInfo + ' Epoch:'+epochInfo
        print(info)

    def rigisterDataGenrated(self):
        self.infoDataGenrated.append(0)

    def DataGenrated(self, threadID):
        self.infoDataGenrated[threadID] += 1

    def setEpoch(self):
        self.epoch += 1
        if self.epoch > self.config['epoch']:
            self.epoch = 1

    def setEposide(self):
        self.eposide += 1


logger = Logger()


def getLogger():
    return logger
