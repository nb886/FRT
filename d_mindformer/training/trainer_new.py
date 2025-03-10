import numpy as np
import torch

import time


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        iter_num=None,
        scheduler=None,
        eval_fns=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        # self.iter_num = iter_num
        self.start_time = time.time()
        self.stop = 0

    def train_iteration(self, num_steps, iter_num, print_logs=False):

        train_losses = []
        logs = dict()
        self.iter_num = iter_num

        train_start = time.time()

        self.model.train()
        if self.stop == 0:
            for _ in range(num_steps):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        # for _ in range(num_steps):
        #     train_loss = self.train_step()
        #     train_losses.append(train_loss)
        #     if self.scheduler is not None:
        #         self.scheduler.step()
        logs["time/training"] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        # if self.iter_num ==1 or self.iter_num == 2 or self.iter_num == 3 or self.iter_num == 5 or self.iter_num == 10 or self.iter_num == 20 :
        if self.iter_num >= 1:
            for eval_fn in self.eval_fns:
                outputs, self.stop = eval_fn(self.model, self.stop)
                for k, v in outputs.items():
                    logs[f"evaluation/{k}"] = v

            logs["time/total"] = time.time() - self.start_time
            logs["time/evaluation"] = time.time() - eval_start
            logs["training/train_loss_mean"] = np.mean(train_losses)
            logs["training/train_loss_std"] = np.std(train_losses)

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            if print_logs:
                print("=" * 80)
                print(f"Iteration {iter_num}")
                for k, v in logs.items():
                    print(f"{k}: {v}")

            return logs
        # for eval_fn in self.eval_fns:
        #     outputs, self.stop = eval_fn(self.model, self.stop)
        #     for k, v in outputs.items():
        #         logs[f"evaluation/{k}"] = v
        #
        # logs["time/total"] = time.time() - self.start_time
        # logs["time/evaluation"] = time.time() - eval_start
        # logs["training/train_loss_mean"] = np.mean(train_losses)
        # logs["training/train_loss_std"] = np.std(train_losses)
        #
        # for k in self.diagnostics:
        #     logs[k] = self.diagnostics[k]
        #
        # if print_logs:
        #     print("=" * 80)
        #     print(f"Iteration {iter_num}")
        #     for k, v in logs.items():
        #         print(f"{k}: {v}")
        #
        # return logs

    def train_step(self):

        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size, self.iter_num
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
            target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
