#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

import numpy as np
from tqdm import tqdm

from dde.data import str_to_mol
from dde.predictor import Predictor


def parse_command_line_arguments():
    """
    Parse the command-line arguments being passed to RMG Py. This uses the
    :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', metavar='FILE',
                        help='Each row should contain reactant_smi, product_smi')

    parser.add_argument('-o', '--out_fname', metavar='FILE',
                        help='Name (without file ending) to save FP arrays to')

    parser.add_argument('-i', '--input', metavar='FILE',
                        help='Path to predictor input file')

    parser.add_argument('-w', '--weights', metavar='H5',
                        help='Path to model weights')

    parser.add_argument('-a', '--architecture', metavar='JSON',
                        help='Path to model architecture (necessary if using uncertainty)')

    parser.add_argument('-ms', '--mean_and_std', metavar='NPZ',
                        help='Saved mean and standard deviation. '
                             'Should be loaded alongside weights if output was normalized during training')

    return parser.parse_args()


def load_data(fpath):
    """Load reactants and products."""
    reactants, products = [], []
    with open(fpath) as f:
        for line in f:
            ls = line.strip().split()  # Separated by spaces
            if ls:
                reactants.append(ls[0])
                products.append(ls[1])
    return reactants, products


def prepare_predictor(input_file, weights_file=None, model_file=None, mean_and_std_file=None):
    predictor = Predictor()
    predictor.load_input(input_file)
    if model_file is not None:
        predictor.load_architecture(model_file)
    predictor.load_parameters(param_path=weights_file, mean_and_std_path=mean_and_std_file)
    return predictor


def get_fps(data_file, input_file, r_file=None, p_file=None,
            weights_file=None, model_file=None, mean_and_std_file=None):

    # load cnn predictor
    predictor = prepare_predictor(input_file, weights_file=weights_file,
                                  model_file=model_file, mean_and_std_file=mean_and_std_file)

    reactants, products = load_data(data_file)
    rfps, pfps = [], []
    for r, p in tqdm(zip(reactants, products)):
        rmol = str_to_mol(r)
        pmol = str_to_mol(p)
        rfps.append(predictor.predict(rmol, fponly=True))
        pfps.append(predictor.predict(pmol, fponly=True))

    np.savez(r_file, *rfps)
    np.savez(p_file, *pfps)


def main():

    args = parse_command_line_arguments()

    data_file = args.data
    out_fname = args.out_fname
    input_file = args.input
    weights_file = args.weights
    model_file = args.architecture
    mean_and_std_file = args.mean_and_std
    r_file = out_fname + '_r.npz'
    p_file = out_fname + '_p.npz'
    get_fps(data_file, input_file, r_file=r_file, p_file=p_file, weights_file=weights_file,
            model_file=model_file, mean_and_std_file=mean_and_std_file)


if __name__ == '__main__':
    main()