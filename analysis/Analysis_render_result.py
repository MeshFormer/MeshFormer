import pandas as pd 
import csv

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


def read_ttsv(file):
    data = pd.read_csv(file, sep='\t', header=None)
    return data.values

def read_fmtsv(file):
    data = pd.read_csv(file, sep=',', header=None)
    facelabel = [int(str_to_bool(i.split('(')[1])) for i in data.values[:, 0]]
    return facelabel

def read_imtsv(file):
    data = pd.read_csv(file, sep=',', header=None)
    facelabel = [int(i.split('(')[1]) for i in data.values[:, 0]]
    return facelabel