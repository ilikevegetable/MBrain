import random
import torch
import numpy as np
from copy import deepcopy
from bisect import bisect_left
import sys
import psutil
import json


def update_logs(logs, log_step, prev_logs=None):
    out = {}
    for key in logs:
        out[key] = deepcopy(logs[key])

        if prev_logs is not None:
            out[key] -= prev_logs[key]
        out[key] /= log_step
    return out


def show_logs(text, logs):
    print("")
    print('-'*50)
    print(text)

    for key in logs:
        if key == "iter":
            continue

        n_predicts = logs[key].shape[0]

        str_steps = ['Step'] + [str(s) for s in range(1, n_predicts + 1)]
        format_command = ' '.join(['{:>16}' for x in range(n_predicts + 1)])
        print(format_command.format(*str_steps))

        str_log = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(format_command.format(*str_log))

    print('-'*50)

def show_downstream_logs(text, logs):
    print("")
    print('-'*50)
    print(text)

    for key in logs:
        if key == "iter":
            continue

        n_predicts = logs[key].shape[0]

        str_log = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        format_command = ' '.join(['{:>16}' for x in range(n_predicts + 1)])
        print(format_command.format(*str_log))

    print('-'*50)


def save_logs(data, path_logs):
    with open(path_logs, 'w') as file:
        json.dump(data, file, indent=2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def ramp_scheduling_function(n_epoch_ramp, epoch):
    if epoch >= n_epoch_ramp:
        return 1
    else:
        return (epoch + 1) / n_epoch_ramp


class SchedulerCombiner:
    r"""
    An object which applies a list of learning rate schedulers sequentially.
    """
    def __init__(self, scheduler_list, activation_step, curr_step=0):
        r"""
        Args:
            - scheduler_list (list): a list of learning rate schedulers
            - activation_step (list): a list of int. activation_step[i]
            indicates at which step scheduler_list[i] should be activated
            - curr_step (int): the starting step. Must be lower than
            activation_step[0]
        """
        if len(scheduler_list) != len(activation_step):
            raise ValueError("The number of scheduler must be the same as "
                             "the number of activation step")
        if activation_step[0] > curr_step:
            raise ValueError("The first activation step cannot be higher than "
                             "the current step.")
        self.scheduler_list = scheduler_list
        self.activation_step = deepcopy(activation_step)
        self.curr_step = curr_step

    def step(self):
        self.curr_step += 1
        index = bisect_left(self.activation_step, self.curr_step) - 1
        for i in reversed(range(index, len(self.scheduler_list))):
            self.scheduler_list[i].step()

    def __str__(self):
        out = "SchedulerCombiner \n"
        out += "(\n"
        for index, scheduler in enumerate(self.scheduler_list):
            out += f"({index}) {scheduler.__str__()} \n"
        out += ")\n"
        return out


def save_checkpoint_ts(ssl_model_state, ssl_criterion_state,
                       replace_criterion_state, timeShift_criterion_state,
                       optimizer_state, best_model_state, best_acc,
                       path_checkpoint):
    state_dict = {"sslModel": ssl_model_state,
                  "sslCriterion": ssl_criterion_state,
                  "replaceCriterion": replace_criterion_state,
                  'timeShiftCriterion': timeShift_criterion_state,
                  "optimizer": optimizer_state,
                  "BestModel": best_model_state,
                  "BestACC": best_acc}

    torch.save(state_dict, path_checkpoint)


def save_checkpoint_newDownstream(ssl_model_state, criterion_state, optimizer_state,
                                  downstreamModel_state, best_downstreamModel_state,
                                  best_model_state, best_criterion_state, best_metric,
                                  path_checkpoint):
    state_dict = {"sslModel": ssl_model_state,
                  "downstreamCriterion": criterion_state,
                  "optimizer": optimizer_state,
                  "BestModel": best_model_state,
                  "BestCriterion": best_criterion_state,
                  "downstreamModel": downstreamModel_state,
                  "BestDownstreamModel": best_downstreamModel_state,
                  "BestMetric": best_metric}

    torch.save(state_dict, path_checkpoint)