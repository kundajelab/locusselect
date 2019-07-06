import argparse
import pandas as pd
import numpy as np 
from locusselect.utils import *

def parse_args():
    parser=argparse.ArgumentParser(description="compute pairwise distances between embeddings")
    parser.add_argument("--embedding_npz")
    parser.add_argument("--clustering_method",default='tsne')
    parser.add_argument("--outf")
    parser.add_argumetn("--output_format",default="svg")
    return parser.parse_args()

def visualize_embeddings(args):
    regions,embeddings,data_type=load_embedding(args.embedding_npz)

def main():
    args=parse_args()
    visualize_embeddings(args)

if __name__=="__main__":
    main()
    
