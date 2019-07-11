# coding:utf-8
import logging
import torch
import torch.nn as nn
from pytorch_mrc.train.trainer import Trainer


class BaseModel(nn.Module):
    def __init__(self, vocab=None, device=None):
        super(BaseModel, self).__init__()

        self.vocab = vocab

        if device is None:
            device = torch.device('cpu')
            logging.warning('No device is assigned, given the default `cpu`')
        if not isinstance(device, torch.device):
            raise TypeError('device must be the instance of `torch.device`, not the instance of `{}`'.format(type(device)))
        self.device = device

        # self.initialized = False
        self.ema_decay = 0

    def __del__(self):
        # TODO
        pass

    def load(self, path, var_list=None):
        # TODO
        # var_list = None returns the list of all saveable variables
        pass
        # self.initialized = True

    def save(self, path, global_step=None, var_list=None):
        # TODO
        pass

    def forward(self, *input):
        raise NotImplementedError

    def compile(self, *input):
        raise NotImplementedError

    def update(self):
        # TODO There are still some problems with logic.
        if not self.training:
            raise Exception("Only in the train mode, you can update the weights")
        if self.optimizer is None:
            raise Exception("The model need to compile!")

        self.optimizer.step()
        # self.optimizer.zero_grad()

    def get_best_answer(self, *input):
        raise NotImplementedError

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=1, episodes=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10, log_every_n_batch=100):
        Trainer.train_and_evaluate(self, self.device, train_generator, eval_generator, evaluator,
                                   epochs=epochs, episodes=episodes,
                                   save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps,
                                   log_every_n_batch=log_every_n_batch)

    def evaluate(self, batch_generator, evaluator):
        Trainer.evaluate(self, self.device, batch_generator, evaluator)

    def inference(self, batch_generator):
        Trainer.inference(self, self.device, batch_generator)
