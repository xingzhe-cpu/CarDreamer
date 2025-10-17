import collections

import numpy as np

from .basics import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._acts = {k: convert(np.zeros((len(self._env),) + v.shape, v.dtype)) for k, v in self._env.act_space.items()}
        self._acts["reset"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._eps_info = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        assert all(len(x) == len(self._env) for x in self._acts.values())
        acts = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
        #根据上一轮policy给出的acts步入环境
        obs, info = self._env.step(acts)
        #类型转换
        obs = {k: convert(v) for k, v in obs.items()}
        info = {k: convert(v) for k, v in info.items()}
        #校验batch大小
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        #调用policy获取动作
        acts, self._state = policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}
        #判断是否最后一个动作，若是则将其mask掉。同时将reset动作设置为True
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()
        #保存新的策略输出
        self._acts = acts
        #将观测和动作合并，形成一个transition
        trns = {**obs, **acts}
        #对于新进入的episode，清空之前的episode数据
        if obs["is_first"].any():
            for i, first in enumerate(obs["is_first"]):
                if first:
                    self._eps[i].clear()
        #将transition和info数据添加到对应的episode中
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            inf = {k: v[i] for k, v in info.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [self._eps_info[i][k].append(v) for k, v in inf.items()]
            [fn(trn, inf, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        #对于结束的episode，调用on_episodes回调函数，便于统计。
        if obs["is_last"].any():
            for i, done in enumerate(obs["is_last"]):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    ep_info = {k: convert(v) for k, v in self._eps_info[i].items()}
                    [fn(ep.copy(), ep_info.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
