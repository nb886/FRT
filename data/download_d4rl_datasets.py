import gym
import numpy as np

import collections
import pickle

import d4rl

# 创建空列表，用于存储数据集
datasets = []

# 遍历不同环境和数据集类型
# for env_name in ['halfcheetah', 'hopper', 'walker2d']:
# 	for dataset_type in ['medium', 'medium-replay', 'expert']:
for env_name in ["ant"]:

    for dataset_type in ["medium"]:
        # 构造环境名称
        name = f"{env_name}-{dataset_type}-v2"
        # 创建环境
        env = gym.make(name)
        # 获取数据集
        dataset = env.get_dataset()

        # 获取数据集中的回报数量
        N = dataset["rewards"].shape[0]
        # 创建空字典，用于存储轨迹数据
        data_ = collections.defaultdict(list)

        # 判断是否使用超时
        use_timeouts = False
        if "timeouts" in dataset:
            use_timeouts = True

        # 初始化轨迹步数
        episode_step = 0
        # 初始化轨迹列表
        paths = []
        # 遍历数据集中的每个样本
        for i in range(N):
            # 获取当前样本的终止标志
            done_bool = bool(dataset["terminals"][i])
            # 判断是否使用超时
            if use_timeouts:
                # 获取当前样本的超时时间
                final_timestep = dataset["timeouts"][i]
            else:
                # 如果不是超时终止，则根据轨迹步数判断是否终止
                final_timestep = episode_step == 1000 - 1
            # 将当前样本的观测、下一个观测、动作、回报、终止标志添加到字典中
            for k in [
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "terminals",
            ]:
                data_[k].append(dataset[k][i])
            # 如果当前样本为终止样本或超时样本，则将当前轨迹数据添加到轨迹列表中，并清空字典
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                # 将字典中的数据转换为numpy数组，并添加到轨迹数据中
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            # 更新轨迹步数
            episode_step += 1

        # 计算每个轨迹的回报总和
        returns = np.array([np.sum(p["rewards"]) for p in paths])
        # 计算轨迹列表中样本数量的总和
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        # 打印样本数量和轨迹回报的统计信息
        print(f"Number of samples collected: {num_samples}")
        print(
            f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
        )

        # 将轨迹列表保存为pickle文件
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(paths, f)
