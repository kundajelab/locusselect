import argparse
import pandas as pd
import numpy as np 
from locusselect.utils import *

def parse_args():
    parser=argparse.ArgumentParser(description="compute pairwise distances between embeddings")
    parser.add_argument("--embedding_npz")
    parser.add_argument("--distance_formula")
    parser.add_argument("--out_hdf5")
    return parser.parse_args()

def compute_embedding_distances(args):
    regions,embeddings,data_type=load_embedding(args.embedding_npz)
    

def main():
    args=parse_args()
    compute_embedding_distances(args)

if __name__=="__main__":
    main()
    
