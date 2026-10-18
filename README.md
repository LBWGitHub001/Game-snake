# 贪吃蛇强化学习 (Snake RL with PPO)

基于 **PyTorch** 和 **PPO (Proximal Policy Optimization)** 深度强化学习算法训练的贪吃蛇智能体。

---

## 一、环境准备

推荐使用 Conda 虚拟环境（如已创建 `RL` 环境）：

```bash
# 激活环境
conda activate RL

# 安装必要依赖（若未安装）
pip install -r requirements.txt
```

---

## 二、运行游戏

### 1. 人类玩家游玩模式 (Human Play)
通过键盘直接体验贪吃蛇小游戏：
```bash
python game.py
```
- **控制按键**：方向键 `[↑]` `[↓]` `[←]` `[→]` 控制移动。
- **重新开始**：死亡后按 `[空格键]` 或 `[R]`。
- **退出游戏**：`[ESC]` 或直接关闭窗口。

---

### 2. 训练强化学习模型 (多线程并行采集 + Offline PPO 批次优化)
```bash
# 启动多线程并行训练（默认开启 8 个游戏线程，极速采集）
python train.py --num-envs 8 --steps-per-env 256 --epochs 4 --total-timesteps 300000

# 可选：导出训练过程中的离线轨迹数据集 (.npz) 用于后续分析与回放
python train.py --num-envs 16 --save-dataset ./Datasets/snake_offline.npz
```

#### 常用训练参数：
- `--num-envs`: 并发游戏线程数（默认 `8`，可设 `8`、`16` 等）。多线程并行开启多个游戏，极大丰富探索状态并成倍提升采样吞吐量。
- `--steps-per-env`: 每个游戏线程采集步数（默认 `256`），一次 Rollout 产生 $8 \times 256 = 2048$ 条转移样本。
- `--epochs`: Offline 数据集批次更新轮次（默认 `4`），对收集的大样本池进行多轮 Mini-batch 深度迭代。
- `--batch-size`: Mini-batch 梯度更新批次大小（默认 `128`）。
- `--total-timesteps`: 总训练步数（默认 `300000`）。
- `--lr`: Adam 优化器学习率（默认 `0.0003`）。
- `--gamma`: 折扣因子 $\gamma$（默认 `0.99`）。
- `--gae-lambda`: 广义优势估计（GAE）因子（默认 `0.95`）。
- `--save-path`: 模型权重保存路径（默认 `./Models/AC.pth`）。
- `--save-dataset`: 可选离线数据集导出路径（如 `./Datasets/offline.npz`）。

---

### 3. 查看与评估训练好的 AI (Evaluation & Visualization)
加载已训练好的权重模型，观察贪吃蛇自主游玩：
```bash
# 评估最新保存的模型
python export.py --model-path ./Models/best_model.pth --episodes 10 --fps 15
```

---

## 三、强化学习算法与架构原理

1. **状态表征 (State Space)**
   - 观测网格为 $12 \times 12$，三通道特征图输入 `(3, 12, 12)`：
     - **Channel 0（蛇头）**：头部位置为 1.0，其余为 0。
     - **Channel 1（蛇身）**：沿颈部到尾部呈指数衰减的梯度数值，保留身体几何形态与蛇身走向。
     - **Channel 2（食物）**：食物所在网格点为 1.0。

2. **动作空间 (Action Space)**
   - 采用相对朝向控制，共 3 种离散动作：
     - `0`: 向左转
     - `1`: 保持直行
     - `2`: 向右转
   - 这种相对坐标设计天然消除了 180 度直接掉头自杀的非法动作，大幅降低了策略搜索空间。

3. **Actor-Critic 网络结构**
   - 采用共享特征提取层（3层保持空间网格分辨率的卷积层 + 共享全连接层）。
   - **Actor Head**：输出 3 个离散动作的未归一化对数概率（Logits），采用 Categorical 分布采样。
   - **Critic Head**：输出状态期望累计收益 $V(s)$。

4. **奖励函数 (Reward Shaping)**
   - 吃到食物：`+10.0`
   - 撞墙或撞自身死亡：`-10.0`
   - 朝食物方向靠近：`+1.0`；远离食物：`-1.0`
   - 防死循环机制：长时间（超过步数上限）未吃到食物判定为饿死并终止游戏。
