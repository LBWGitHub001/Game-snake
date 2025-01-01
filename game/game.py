import threading
from threading import Thread, Lock

import pygame
import torch
import torch.nn as nn
from Logger import getLogger
import copy
from game.gameEngine import GameEngine
import queue
import time

cvParse = threading.Condition()
cvUpdate = threading.Condition()


## 本代码中使用只有一个变量的list传参表示引用传递，请用者记住

class DataBlock:
    def __init__(self, id, data):
        self.id = id
        self.data = data


class DataBlockThread:
    def __init__(self, id, config, dataModel, dataSelector, device, render):
        self.config = config
        self.id = id
        self.DataNN = dataModel  # 线程中运行的数据采集网络(DataNN)
        self.optimer = torch.optim.Adam(self.DataNN.parameters(), lr=1e-3)
        self.tempMem = list()  # 线程数据缓存
        self.game = GameEngine(config)  # 游戏进程
        self.stop = False  # 停止符，在适当时间停止游戏
        self.dataSelector = dataSelector[0]
        self.runThread = None  # 游戏和数据收集线程，使用createThread函数正式创建
        self.timestep = 0
        self.havenRun = False  #
        self.device = device
        self.render = render

        self.states = None
        self.actions = None
        self.awards = None

        # 图形显示
        self.screen = None

    def setTimestep(self, timestep):  # 为程序设置一个eposide
        self.timestep = timestep

    def createThread(self):  # 创建游戏线程和数据收集模型
        def thread_func():
            getLogger().rigisterDataGenrated()
            iterCount = 0
            if self.render:
                pygame.init()
                self.screen = pygame.display.set_mode((self.config['game']['xpx'] * self.config['game']['blockSize'],
                                                       self.config['game']['ypx'] * self.config['game']['blockSize']))
                pygame.display.set_caption('游戏显示实例')
                self.screen.fill((255, 255, 255))
            while True:
                while not self.stop:
                    state = self.game.getState().to(self.device)  # 从game中读取当前的状态
                    state.unsqueeze_(dim=0)
                    action = self.DataNN.action_layer(state).to(self.device)  # 这里state和action都应该是和NN匹配的状态
                    self.game.SetAction(action)
                    award = self.game.getAward().to(self.device)  # 从这里获取本次状态的奖励
                    if self.states == None:  # 合并states
                        self.states = state
                    else:
                        self.states = torch.cat((self.states, state), dim=0)

                    if self.awards == None:
                        self.awards = award
                    else:
                        self.awards = torch.cat((self.awards, award), dim=0)

                    _, arg = torch.max(action, dim=1)
                    if self.actions == None:
                        self.actions = arg
                    else:
                        self.actions = torch.cat((self.actions, arg), dim=0)
                    iterCount += 1  # 数据数+1

                    if self.render:  # 图形显示
                        size = self.config['game']['blockSize']
                        for j in range(self.config['game']['xpx']):
                            for i in range(self.config['game']['ypx']):
                                r = state[0][0][i][j].item()
                                g = state[0][1][i][j].item()
                                b = state[0][2][i][j].item()
                                color = (r, g, b)
                                x = i * size
                                y = j * size
                                w = size
                                h = size
                                block = (x, y, w, h)
                                pygame.draw.rect(self.screen, color, block)
                            pygame.display.flip()
                    if iterCount >= self.timestep:  # 数据收集够了之后，将缓存中的数据写入数据管理
                        # print(self.id)  # Debug点
                        dataBlock = DataBlock(self.id, (self.states, self.actions, self.awards))  # 构筑一个数据包
                        # print(len(self.tempMem)) # Debug
                        self.dataSelector.write(dataBlock)  # 向DataSelector中传入一个数据包
                        self.tempMem.clear()  # 清空缓存中的数据
                        iterCount -= self.timestep
                        getLogger().DataGenrated(self.id)
                        # print('Thread' + str(self.id), 'has completed data selection once!')

        self.runThread = Thread(target=thread_func)  # 创建一个线程
        return self.runThread

    def run(self):
        self.runThread.start()  # 启动一个线程

    def update(self, parser):  # 更新参数的数据
        self.stop = True
        self.DataNN.load_state_dict(parser)
        self.stop = False


## 数据收集器，原子地收集数据
class DataSelector:
    def __init__(self):
        self.trainQueue = queue.Queue()  # 更新网络的任务队列
        self.updateQueue = queue.Queue()  # 更新参数的任务队列
        # self.trainNN = trainModel[0]  # 待训练更新参数的网络模型
        self.busy = False  # 防止数据竞态，原子标识位
        self.dataModelList = None

    def setDataModelList(self, dataModelList):
        self.dataModelList = dataModelList

    def write(self, externData):  # 从缓存中读取一个参数
        cvParse.acquire()
        self.busy = True
        self.trainQueue.put(externData)  # 读取一个数据包组
        # print('writed')
        self.busy = False
        cvParse.notify()
        cvParse.release()
        time.sleep(1)

    def updateModel(self, modelId, parser):  # 更新一个模型的参数
        self.busy = True
        dateModel = self.dataModelList[modelId]
        dateModel.stop = False  # 暂停程序进行的任务
        dateModel.update(parser)
        self.busy = False

    def read(self):
        data = self.trainQueue.get()
        # print('read')
        return data

    def empty(self):
        return self.trainQueue.empty()
