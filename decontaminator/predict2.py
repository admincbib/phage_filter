#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Grigorii Sukhorukov, Macha Nikolski
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
# loglevel : 0 all printed, 1 I not printed, 2 I and W not printed, 3 nothing printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import fire
import yaml
import tensorflow as tf
import numpy as np
from Bio import SeqIO
import pandas as pd
from utils import preprocess as pp
from pathlib import Path
from joblib import load
import predict as pr


def predict2(test_ds, weights1, weights2, out_path):
    """filters out contaminant contigs from the fasta file. Pipeline for 2 filters.

    test_ds: path to the input file with contigs in fasta format (str or list of str)
    weights1: path to the folder containing weights for the first filtering
    weights2: path to the folder containing weights for the first filtering
    out_path: path to the folder to store predictions (str)
    return_viral: whether to return contigs annotated as viral in separate fasta file (True/False)
    """

    test_ds = test_ds
    if isinstance(test_ds, list):
        pass
    elif isinstance(test_ds, str):
        test_ds = [test_ds]
    else:
        raise ValueError('test_ds was incorrectly assigned in the config file')

    assert Path(test_ds[0]).exists(), f'{test_ds[0]} does not exist'
    assert Path(weights1).exists(), f'{weights1} does not exist'
    assert Path(weights2).exists(), f'{weights2} does not exist'
    Path(out_path).mkdir(parents=True, exist_ok=True)

    pr.predict(
        test_ds=test_ds,
        weights=weights1,
        out_path=out_path,
        return_viral=True,
        limit=0,
    )
    test_ds_2 = [Path(out_path, f"{Path(ts).stem}_viral.fasta") for ts in test_ds]
    pr.predict(
        test_ds=test_ds_2,
        weights=weights2,
        out_path=out_path,
        return_viral=True,
        limit=0,
    )


def predict2_config(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    predict2(
        test_ds=cf['predict2']['test_ds'],
        weights1=cf['predict2']['weights1'],
        weights2=cf['predict2']['weights2'],
        out_path=cf['predict2']['out_path'],
    )


if __name__ == '__main__':
    fire.Fire(predict2_config)
