import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import compute_stats_by_type, named_transpose, transpose_list, rms

def plot_train_results(exp,train_results,test_results,prms):
    pass  # fill in your plotting logic

def plot_dropout_results(exp,dropout_results,dropout_parameters,prms,dropout_type="nodes"):
    pass  # fill in your dropout plots

def plot_eigenfeatures(exp,results,prms):
    pass  # fill in your eigenfeature plots

def plot_adversarial_results(exp,eigen_results,adversarial_results,prms):
    pass  # fill in your adversarial analysis plots