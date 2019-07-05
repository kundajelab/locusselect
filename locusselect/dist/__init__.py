import argparse
import pandas as pd
import numpy as np 

def parse_args():
    parser=argparse.ArgumentParser(description="compute pairwise distances between embeddings")
    parser.add_argument("--embedding_npz")
    parser.add_argument("--distance_formula")
    parser.add_argument("--out_hdf5")
    return parser.parse_args()

def compute_embedding_distances(args):
    data=np.load(args.embedding_npz)
    regions=data['bed_entries']
    embeddings=data['embeddings']
    

def main():
    args=parse_args()
    compute_embedding_distances(args)

if __name__=="__main__":
    main()
    
