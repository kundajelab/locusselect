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

def load_narrowPeak_file(data_path,start_row,num_rows):

    data=pd.read_csv(data_path,header=None,sep='\t',index_col=[0,1,2])
    if num_rows is None:
        return data
    else:
        end_row=min(start_row+num_rows,data.shape[0])
        data=data.iloc[start_row:end_row]
        return data

#use wrappers for keras Sequence generator class to allow batch shuffling upon epoch end
class DataGenerator(Sequence):
    def __init__(self,data_path,ref_fasta,batch_size=128,center_on_summit=False,center_on_bed_interval=False,flank=None,expand_dims=False,start_row=0,num_rows=None):
        self.lock = threading.Lock()        
        self.batch_size=batch_size
        #open the reference file
        self.ref_fasta=ref_fasta
        self.data=load_narrowPeak_file(data_path,start_row,num_rows)
        self.data_index=self.data.index.values  
        self.indices=np.arange(self.data.shape[0])
        self.center_on_summit=center_on_summit
        self.center_on_bed_interval=center_on_bed_interval
        self.flank=flank
        self.expand_dims=expand_dims
        self.start_row=start_row
        self.num_rows=num_rows
        
        
    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self,idx):
        with self.lock:
            ref=pysam.FastaFile(self.ref_fasta)
            self.ref=ref
            return self.get_batch(idx) 

    def get_bed_entries_from_inds(self,inds):        
        if self.center_on_summit is True:
            assert self.center_on_bed_interval == False 
            #get the specified flank around the peak summit
            bed_rows=self.data.iloc[inds]
            summit_col=max(bed_rows.columns)
            bed_entries=[]
            for index,row in bed_rows.iterrows():
                start_pos=max([0,index[1]+row[summit_col]-self.flank])
                end_pos=max([2*self.flank,index[1]+row[summit_col]+self.flank])
                chrom=index[0] 
                bed_entries.append([chrom,start_pos,end_pos])
            return bed_entries
        
        elif self.center_on_bed_interval is True:
            assert self.center_on_summit == False
            #get the specified flank around the peak center
            bed_rows=self.data.iloc[inds]
            bed_entries=[]
            for index,row in bed_rows.iterrows():
                peak_center=int(math.floor(0.5*(index[1]+index[2])))
                chrom=index[0]
                start_pos=max([0,peak_center-self.flank])
                end_pos=max([2*self.flank,peak_center+self.flank])
                bed_entries.append([chrom,start_pos,end_pos])
            return bed_entries
        else:
            #return the chrom,start, end columns from the narrowPeak file
            return self.data.index[inds]
                
    def get_batch(self,idx):
        #get seq positions
        inds=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        bed_entries=self.get_bed_entries_from_inds(inds)
        #get sequences
        seqs=[self.ref.fetch(i[0],i[1],i[2]) for i in bed_entries]
        corrected_seqs=[]
        for seq in seqs:
            try:
                if len(seq)<2*self.flank:
                    delta=2*self.flank-len(seq)
                    seq=seq+"N"*delta
            except:
                seq="N"*2*self.flank
            corrected_seqs.append(seq)
        #one-hot-encode the fasta sequences 
        seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in corrected_seqs])
        x_batch=seqs
        if self.expand_dims==True:
            x_batch=np.expand_dims(x_batch,1)
        indices=self.data_index[inds]
        return x_batch,indices
    
    
