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
from Bio import SeqIO
import random
import numpy as np
from utils import preprocess as pp
from sklearn.utils import shuffle
import h5py
from pathlib import Path


def prepare_ds_nn(
        path_virus,
        path_other,
        out_path,
        fragment_length,
        random_seed=None,
):
    """
    This is a function for the example dataset preparation.
    If random seed is not specified it will be generated randomly.
    """

    if random_seed is None:
        random.seed(a=random_seed)
        random_seed = random.randrange(1000000)
    print(f'starting generation using random seed {random_seed}')
    random.seed(a=random_seed)

    v_encoded, v_encoded_rc, v_labs, v_seqs, v_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_virus, label='virus', label_int=1, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length / 2), max_gap=0.05,)

    ph_encoded, ph_encoded_rc, ph_labs, ph_seqs, ph_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_other, fragment_length=fragment_length,
        n_frags=v_n_frags, label='other', label_int=0, random_seed=random.randrange(1000000))

    assert v_n_frags == ph_n_frags

    all_encoded = np.concatenate([v_encoded, ph_encoded])
    all_encoded_rc = np.concatenate([v_encoded_rc, ph_encoded_rc])
    all_labs = np.concatenate([v_labs, ph_labs])

    # adding reverse complement
    all_encoded = np.concatenate((all_encoded, all_encoded_rc))
    all_encoded_rc = np.concatenate((all_encoded_rc, all_encoded))
    all_labs = np.concatenate((all_labs, all_labs))

    # saving one-hot encoded fragments
    pp.storing_encoded(all_encoded, all_encoded_rc, all_labs,
                       Path(out_path, f"encoded_train_{fragment_length}.hdf5"))


def prepare_ds(path_virus, path_other, out_path, random_seed):
    assert Path(path_virus).exists(), f'{path_virus} does not exist'
    assert Path(path_other).exists(), f'{path_other} does not exist'
    Path(out_path).mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        prepare_ds_nn(
            path_virus=path_virus,
            path_other=path_other,
            out_path=out_path,
            fragment_length=l_,
            random_seed=random_seed,
        )
    print(f"NN datasets are stored in {out_path}")


def prepare_ds_config(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    prepare_ds(
        path_virus=cf["prepare_ds"]["path_virus"],
        path_other=cf["prepare_ds"]["path_other"],
        out_path=cf["prepare_ds"]["out_path"],
        random_seed=cf["prepare_ds"]["random_seed"],
    )


if __name__ == '__main__':
    fire.Fire(prepare_ds_config)
