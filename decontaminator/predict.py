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
from models import model_10
from joblib import load


def predict_nn(ds_path, nn_weights_path, length, batch_size=256):
    """
    Breaks down contigs into fragments
    and uses pretrained neural networks to give predictions for fragments
    """
    try:
        seqs_ = list(SeqIO.parse(ds_path, "fasta"))
    except FileNotFoundError:
        raise Exception("test dataset was not found. Change ds variable")

    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_vir_10": [],
        "pred_phage_10": [],
    }
    if not seqs_:
        raise ValueError("All sequences were smaller than length of the model")
    test_fragments = []
    test_fragments_rc = []
    for seq in seqs_:
        fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8,
                                                     sl_wind_step=int(length / 2))
        test_fragments.extend(fragments_)
        test_fragments_rc.extend(fragments_rc)
        for j in range(len(fragments_)):
            out_table["id"].append(seq.id)
            out_table["length"].append(len(seq.seq))
            out_table["fragment"].append(j)
    test_encoded = pp.one_hot_encode(test_fragments)
    test_encoded_rc = pp.one_hot_encode(test_fragments_rc)
    model = model_10.model(length)
    model.load_weights(Path(nn_weights_path, f"model_{length}.h5"))
    prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
    out_table['pred_vir'].extend(list(prediction[..., 1]))
    out_table['pred_other'].extend(list(prediction[..., 0]))
    print('Exporting predictions to csv file')
    df = pd.DataFrame(out_table)
    df['NN_decision'] = np.where(df['pred_vir'] > df['pred_other'], 'virus', 'other')
    return df


def predict_contigs(df):
    """
    Based on predictions of predict_rf for fragments gives a final prediction for the whole contig
    """
    df = (
        df.groupby(["id", "length", 'NN_decision'], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    df = df.reset_index()
    df = df.reindex(['length', 'id', 'virus', 'other',], axis=1)
    df['decision'] = np.where(df['virus'] >= df['other'], 'virus', 'other')
    df = df.sort_values(by='length', ascending=False)
    df = df.loc[:, ['length', 'id', 'virus', 'other', 'decision']]
    df = df.rename(columns={'virus': '# viral fragments', 'phage': '# other fragments',})
    df['# viral / # total'] = (df['# viral fragments'] / (
                df['# viral fragments'] + df['# other fragments'])).round(3)
    df['# viral / # total * length'] = df['# viral / # total'] * df['length']
    df = df.sort_values(by='# viral / # total * length', ascending=False)
    return df


def predict(test_ds, weights, out_path, return_viral, limit):
    """Predicts viral contigs from the fasta file

    test_ds: path to the input file with contigs in fasta format (str or list of str)
    weights: path to the folder containing weights for NN and RF modules trained on 500 and 1000 fragment lengths (str)
    out_path: path to the folder to store predictions (str)
    return_viral: whether to return contigs annotated as viral in separate fasta file (True/False)
    limit: Do predictions only for contigs > l. We suggest l=750. (int)
    """

    test_ds = test_ds
    if isinstance(test_ds, list):
        pass
    elif isinstance(test_ds, str):
        test_ds = [test_ds]
    else:
        raise ValueError('test_ds was incorrectly assigned in the config file')

    assert Path(test_ds[0]).exists(), f'{test_ds[0]} does not exist'
    assert Path(weights).exists(), f'{weights} does not exist'
    assert isinstance(limit, int), 'limit should be an integer'
    Path(out_path).mkdir(parents=True, exist_ok=True)

    for ts in test_ds:
        dfs_fr = []
        dfs_cont = []
        for l_ in 500, 1000:
            print(f'starting prediction for {Path(ts).name} for fragment length {l_}')
            df = predict_nn(
                ds_path=ts,
                nn_weights_path=weights,
                length=l_,
            )
            dfs_fr.append(df)
            df = predict_contigs(df)
            dfs_cont.append(df)
            print('prediction finished')
        df_500 = dfs_fr[0][(dfs_fr[0]['length'] >= 750) & (dfs_fr[0]['length'] < 1500)]
        df_1000 = dfs_fr[1][(dfs_fr[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_fr = Path(out_path, f"{Path(ts).stem}_predicted_fragments.csv")
        df.to_csv(pred_fr)

        df_500 = dfs_cont[0][(dfs_cont[0]['length'] >= 750) & (dfs_cont[0]['length'] < 1500)]
        df_1000 = dfs_cont[1][(dfs_cont[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_contigs = Path(out_path, f"{Path(ts).stem}_predicted.csv")
        df.to_csv(pred_contigs)

        if return_viral:
            viral_ids = list(df[df["decision"] == "virus"]["id"])
            seqs_ = list(SeqIO.parse(ts, "fasta"))
            viral_seqs = [s_ for s_ in seqs_ if s_.id in viral_ids]
            SeqIO.write(viral_seqs, Path(out_path, f"{Path(ts).stem}_viral.fasta"), 'fasta')



def predict_config(config):
    with open(config, "r") as yamlfile:
        cf = yaml.load(yamlfile, Loader=yaml.FullLoader)
    predict(
        test_ds=cf['predict']['out_path'],
        weights=cf['predict']['weights'],
        out_path=cf['predict']['out_path'],
        return_viral=cf['predict']['return_viral'],
        limit=cf['predict']['limit'],
    )


if __name__ == '__main__':
    fire.Fire(predict_config)
