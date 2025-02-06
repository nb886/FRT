import numpy as np
import torch
import torch.nn as nn
import transformers

from scipy.stats import norm
# 导入mindformers模块
# import mindformers

import torch.nn.functional as F

from .model import TrajectoryModel
from .trajectory_gpt2 import GPT2Model

# from mindformers.models.gpt2.gpt2 import GPT2Model
# from gpt2.gpt2 import GPT2Model


class DecisionMindformer(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        # config = mindformers.models.gpt2.gpt2.GPT2Config(               # GPT2的参数
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs,
            output_attentions=True
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # self.transformer = GPT2Model(config)
        self.mindformer = GPT2Model(config)
        # get attention mask
        # self.att_mask = att_mask

        # get cross attention mask
        # self.cross_att_mask = cross_att_mask

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.disembed_reward = torch.nn.Linear(hidden_size, 1)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.predict_reward = nn.Sequential(
            *([nn.Linear(hidden_size, 1)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(
        self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)  #
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings    加入时间特征
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings
        

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (
                returns_embeddings,
                state_embeddings,
                action_embeddings,
                rewards_embeddings,
            ),
            dim=1,
        )

        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            batch_size, 4 * seq_length, self.hidden_size
        )
        stacked_inputs = self.embed_ln(stacked_inputs)
        # print('stacked_inputs=',stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack(
                (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
            )
            .permute(0, 2, 1)
            .reshape(batch_size, 4 * seq_length)
        )
        # print(stacked_inputs.dtype)
        # print(stacked_inputs.shape)
        # print(stacked_inputs)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.mindformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )
        # print("att", attention_mask.shape)
        # print("reward", rewards_embeddings.shape)
        attention_mask=attention_mask.to(dtype=state_embeddings.dtype)
        # print("att2", attention_mask.shape)
        broadcasted_mask = attention_mask.unsqueeze(-1)
        # print("att3", broadcasted_mask.shape)
        broadcasted_mask = self.gaussian_blur(broadcasted_mask)

        reward_disembed = self.disembed_reward(rewards_embeddings)
        reward_merge = reward_disembed * broadcasted_mask
        # reward_merge = self.moving_average(reward_merge)

        # print("merge", reward_merge.shape)

        x = transformer_outputs["last_hidden_state"]
        # merge output of transformer with state_embeddings
        # x_merge = torch.cat([x, reward_merge], dim=-1)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)
        
        # reshape x_merge so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(
            0, 2, 1, 3
        )

        # get predictions
        return_preds = self.predict_return(
            x[:, 1]
        )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 2]
        )  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        reward_preds = self.predict_reward(
            x[:, 1]
        )  # predict next action given state    [:,2]初测效果不好
        # 5- 1211  4-2211   3-2212


        # print("reward preds", reward_preds.shape)
        # print("reward preds", reward_preds)
        # print("att3", broadcasted_mask.shape)
        # print("att3", broadcasted_mask)
        # reward_merge = reward_preds * broadcasted_mask
        # print("reward_merge", reward_merge.shape)
        # print("reward_merge", reward_merge)
        # return state_preds, action_preds, return_preds
        return state_preds, action_preds, return_preds, reward_merge

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            rewards = rewards[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            rewards = torch.cat(
                [
                    torch.zeros(
                        (rewards.shape[0], self.max_length - rewards.shape[1], 1),
                        device=rewards.device,
                    ),
                    rewards,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        # _, action_preds_a, return_preds, reward_preds = self.forward(
        #     states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask)
        reward_preds_a = self.get_reward(
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
        )

        _, action_preds_b, reward_preds_a, reward_preds = self.forward(
            states,
            actions,
            reward_preds_a,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
        )

        # _, action_preds_a, return_preds, reward_preds = self.forward(
        #     states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask)
        # reward_preds_a = self.get_reward(states,action_preds_a, rewards, returns_to_go, timesteps, attention_mask=attention_mask)
        #
        # _, action_preds_b, return_preds, reward_preds = self.forward(
        #     states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask)
        # reward_preds_b = self.get_reward(states,action_preds_b, rewards, returns_to_go, timesteps, attention_mask=attention_mask)

        # if reward_preds_a > reward_preds_b:
        #     action_preds = action_preds_a
        # else:
        #     action_preds = action_preds_b

        action_preds = action_preds_b

        # print(action_preds_a[0,-1], action_preds_b[0,-1])
        return action_preds[0, -1]

    def get_reward(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this modelh,

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            rewards = rewards[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            rewards = torch.cat(
                [
                    torch.zeros(
                        (rewards.shape[0], self.max_length - rewards.shape[1], 1),
                        device=rewards.device,
                    ),
                    rewards,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, reward_preds = self.forward(
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
        )

        return reward_preds[0, -1]
    def moving_average(self,sequence_tensor) :
        """
        对给定的一维序列张量进行移动平均平滑处理。

        Args:
            sequence_tensor (torch.Tensor): 形状为 [batch_size, sequence_length, 1] 的一维序列张量。

        Returns:
            torch.Tensor: 经过移动平均平滑处理后的序列张量，形状与输入相同。
        """
        batch_size, sequence_length, _ = sequence_tensor.shape
        window_size = 5
        assert window_size > 0 and window_size <= sequence_length

        # 初始化输出张量
        smoothed_sequence = torch.zeros_like(sequence_tensor)

        # 计算移动窗口内元素的平均值
        for i in range(window_size // 2, sequence_length - window_size // 2):
            window = sequence_tensor[:, i - window_size // 2 : i + window_size // 2 + 1, 0]
            smoothed_sequence[:, i, 0] = window.mean(dim=1)

        # 处理边界情况：对窗口覆盖不到的部分，使用尽可能多的元素计算平均值
        for i in range(window_size // 2):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2 + 1)
            window = sequence_tensor[:, start:end, 0]
            smoothed_sequence[:, i, 0] = window.mean(dim=1)

        for i in range(sequence_length - window_size // 2, sequence_length):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2 + 1)
            window = sequence_tensor[:, start:end, 0]
            smoothed_sequence[:, i, 0] = window.mean(dim=1)

        return smoothed_sequence



    def generate_gaussian_kernel(self,window_size, sigma=1.0):
        """Generate a 1D Gaussian kernel."""
        kernel_range = np.arange(window_size)
        kernel = norm.pdf(kernel_range, loc=window_size // 2, scale=sigma)
        kernel /= kernel.sum()  # Normalize to sum to 1
        return torch.from_numpy(kernel).float()

    def gaussian_blur(self, sequence_tensor, window_size=3, sigma=1.0):
        print(sequence_tensor.shape)
        """Apply Gaussian blur to a 1D sequence tensor."""
        batch_size, sequence_length, _ = sequence_tensor.shape
        gaussian_kernel = self.generate_gaussian_kernel(window_size, sigma)
        gaussian_kernel = gaussian_kernel.reshape(1, 1, -1)  # Expand to 3D tensor
        print(gaussian_kernel.shape)
        print(sequence_tensor.shape)

        # Perform 1D convolution with the Gaussian kernel
        blurred_sequence = F.conv1d(sequence_tensor,
                                    gaussian_kernel,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    groups=1)

        return blurred_sequence



