# coding:utf-8
import torch
import torch.nn as nn
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict
from pytorch_mrc.train.trainer import Trainer


class BaseModel(nn.Module):
    def __init__(self, vocab=None):
        super(BaseModel, self).__init__()

        self.vocab = vocab
        self.initialized = False
        self.ema_decay = 0

    def __del__(self):
        # TODO
        pass

    def load(self, path, var_list=None):
        # TODO
        # var_list = None returns the list of all saveable variables
        pass
        self.initialized = True

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

        if self.optimizer is not None:
            self.optimizer.step()
            # self.optimizer.zero_grad()
        else:
            raise Exception("The model need to compile!")

    def get_best_answer(self, *input):
        raise NotImplementedError

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=1, episodes=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            pass

        Trainer.train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=epochs,
                                    episodes=episodes,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps)

    def evaluate(self, batch_generator, evaluator):
        Trainer.evaluate(self, batch_generator, evaluator)

    def inference(self, batch_generator):
        Trainer.inference(self, batch_generator)
