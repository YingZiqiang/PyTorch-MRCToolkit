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
        self.vocab = vocab

        # sess_conf = tf.ConfigProto()
        # sess_conf.gpu_options.allow_growth = True
        # self.session = tf.Session(config=sess_conf)
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

    def update(self, *input):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            raise Exception("The model need to compile!")

    def get_best_answer(self, *input):
        raise NotImplementedError

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        Trainer._train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=epochs,
                                    eposides=eposides,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps)

    def evaluate(self, batch_generator, evaluator):
        Trainer._evaluate(self, batch_generator, evaluator)

    def inference(self, batch_generator):
        Trainer._inference(self, batch_generator)
