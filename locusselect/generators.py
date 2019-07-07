from keras.utils import Sequence
import pandas as pd
import numpy as np
import math 
import pysam
import threading
import pdb 

ltrdict = {'a':[1,0,0,0],
           'c':[0,1,0,0],
           'g':[0,0,1,0],
           't':[0,0,0,1],
           'n':[0,0,0,0],
           'A':[1,0,0,0],
           'C':[0,1,0,0],
           'G':[0,0,1,0],
           'T':[0,0,0,1],
           'N':[0,0,0,0]}

def load_narrowPeak_file(data_path):
    data=pd.read_csv(data_path,header=None,sep='\t',index_col=[0,1,2])
    return data 


#use wrappers for keras Sequence generator class to allow batch shuffling upon epoch end
class DataGenerator(Sequence):
    def __init__(self,data_path,ref_fasta,batch_size=128,center_on_summit=False,flank=None,expand_dims=False):
        self.lock = threading.Lock()        
        self.batch_size=batch_size
        #open the reference file
        self.ref_fasta=ref_fasta
        self.data=load_narrowPeak_file(data_path)
        self.data_index=self.data.index.values  
        self.indices=np.arange(self.data.shape[0])
        self.center_on_summit=center_on_summit
        self.flank=flank
        self.expand_dims=expand_dims
        
        
    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            ref=pysam.FastaFile(self.ref_fasta)
            self.ref=ref
            return self.get_batch(idx) 

    def get_bed_entries_from_inds(self,inds):
        if self.center_on_summit==False:
            #return the chrom,start, end columns from the narrowPeak file
            return self.data.index[inds]
        else:
            #get the specified flank around the peak summit
            bed_rows=self.data.iloc[inds]
            summit_col=max(bed_rows.columns)
            bed_entries=[]
            for index,row in bed_rows.iterrows():
                bed_entries.append([index[0],
                                    index[1]+row[summit_col]-self.flank,
                                    index[1]+row[summit_col]+self.flank])
            return bed_entries
        
    def get_batch(self,idx):
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.get_bed_entries_from_inds(inds)
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
        x_batch=seqs
        if self.expand_dims==True:
            x_batch=np.expand_dims(x_batch,1)
        indices=self.data_index[inds]
        return x_batch,indices
    
    
