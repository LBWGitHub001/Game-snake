# 贪吃蛇小游戏
## 运行game.py即可开始游戏
> 使用方向键上下左右可以控制蛇的运动方向，去捕食更多食物，变的更长吧

## 模型训练
### 原理
> PPO算法  
> 使用两套神经网络(CNN,全连接)前者作为Actor，后者作为Critic  
> 使用CNN来决策当前状态的动作，使用全连接网络来调整学习任务难度

### 安装cuda[Linux-Ubuntu]
cuda是Nvidia开发的帮助开发人员更高的利用GPU进行并行计算的框架，对于深度学习至关重要，可以起到加速模型的训练和推理
>1. 检查本机的显卡驱动是否安装好`nvidia-smi`如果显示信息说明已经安装完成
>2. 首先进入[Nvidia Cuda官网](https://developer.nvidia.com/cuda-toolkit)下载安装包这里推荐下载
    cuda的11.8版本，推荐runfile模式安装  
    1 `wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run`
    2 `sudo sh cuda_11.8.0_520.61.05_linux.run`
>3. 安装完成后不要关闭终端，按照终端上的提示将库写入环境变量~/.bashrc
    ?4. 检查是否安装成功 `nvcc -V` 如出现编译器版本号则说明成功
### 安装cudnn
cuDNN是Nvidia开发的模型推理加速的拓展库，对于模型的部署至关重要
>1. 前往[Nvidia cuDNN官网](https://developer.nvidia.com/cudnn)下载最新版本即可
>2. 注意此时不需要再次安装cuda，因此下面的两条命令不需要执行
### 安装Anaconda3
Anaconda是一个Python虚拟环境管理器，其中内置了大量Python模块，可以方便的管理模块和虚拟环境
>1. [Anaconda官网](https://www.anaconda.com/download/)下载，注意也要添加到环境变量中
### 安装PyTorch
PyTorch是现在流行的神经网络搭建架构，可以帮助开发人员快速实现神经网络的搭建
>1. 从[官网下载库](https://download.pytorch.org/whl/torch_stable.html)中下载，与cuda版本对应，如果你的cuda版本是11.8，python是3.8，那么应该安装 **[torch-2.0.0+cu118-cp38-cp38-linux_x86_64.whl]**
>2. 下载torchvision，同样版本要对应，不再赘述

## 训练
1. `python train.py`
### 可选参数
>1. [--timestep] 收集多少数据之后开始学习(默认 2000)
>2. [--episodes] 玩多少遍游戏(默认 50000)
>3. [--lr] 学习率(默认 0.03)
>4. [--gamma] 折扣因子(默认 0.99)  
>5. [--save-path] 模型保存路径(默认 './Models/AC.pth')  
>6. [--num-game] 游戏线程(默认 1)
>7. [--epochs] 一次数据训练次数(默认 4)
>8. [--seed] 随机种子,默认为None(默认 None)
>9. [--eps] PPO2算法中的界(默认 0.2)
>10. [--bate1] Adam动量的位置惯性系数1(默认 0.2)
>11. [--bate2] Adam动量的速度惯性系数2(默认 0.2)
>12. [--render] 是否要显示过程(默认 False,指定即为true)


## 调试
1. `python export.py`
### 可选参数
>1. [--save-path]模型存储位置(默认 './Models/AC.pth')
>2. [--seed] 随机种子(默认 None)
>3. [--render] 是否要显示过程(默认 True,指定即为False)
2. 奖励函数调整
>1. GameAPI.get_reward()函数  
> 通过前后两帧的区别对这一步操作进行打分，最关键的步骤，需要更改

