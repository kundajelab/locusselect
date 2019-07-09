import itertools
import argparse 
import numpy as np
import tensorflow as tf 
import pdb

#import dependencies for locusselect 
from locusselect.utils import *
from locusselect.config import args_object_from_args_dict

def parse_args():
    parser=argparse.ArgumentParser(description="compute gapped kmer embeddings")
    parser.add_argument("--importance_score_files",nargs="+")
    parser.add_argument("--kmer_len",type=int,help="length of kmer")
    parser.add_argument("--num_gaps",type=int,help="number of allowed gaps in the kmer")
    parser.add_argument("--alphabet_size",type=int,default=4)
    parser.add_argument("--outf",nargs="*",default=None)
    parser.add_argument("--batch_size",type=int,default=1000)
    return parser.parse_args() 
    
def run_function_in_batches(func,
                            input_data_list,
                            learning_phase=None,
                            batch_size=10,
                            progress_update=1000,
                            multimodal_output=False):
    #func has a return value such that the first index is the
    #batch. This function will run func in batches on the inputData
    #and will extend the result into one big list.
    #if multimodal_output=True, func has a return value such that first
    #index is the mode and second index is the batch
    assert isinstance(input_data_list, list), "input_data_list must be a list"
    #input_datas is an array of the different input_data modes.
    to_return = [];
    i = 0;
    while i < len(input_data_list[0]):
        if (progress_update is not None):
            if (i%progress_update == 0):
                print("Done",i)
        func_output = func(([x[i:i+batch_size] for x in input_data_list]
                                +([] if learning_phase is
                                   None else [learning_phase])
                        ))
        if (multimodal_output):
            assert isinstance(func_output, list),\
             "multimodal_output=True yet function return value is not a list"
            if (len(to_return)==0):
                to_return = [[] for x in func_output]
            for to_extend, batch_results in zip(to_return, func_output):
                to_extend.extend(batch_results)
        else:
            to_return.extend(func_output)
        i += batch_size
    return to_return

def get_session():
    try:
        #use the keras session if there is one
        import keras.backend as K
        return K.get_session()
    except:
        #Warning: I haven't really tested this behaviour out...
        global _SESS 
        if _SESS is None:
            print("MAKING A SESSION")
            _SESS = tf.Session()
            _SESS.run(tf.global_variables_initializer()) 
        return _SESS

def compile_func(inputs, outputs):
    if (isinstance(inputs, list)==False):
        print("Wrapping the inputs in a list...")
        inputs = [inputs]
    assert isinstance(inputs, list)
    def func_to_return(inp):
        if len(inp) > len(inputs) and len(inputs)==1:
            print("Wrapping the inputs in a list...")
            inp = [inp]
        assert len(inp)==len(inputs),\
            ("length of provided list should be "
             +str(len(inputs))+" for tensors "+str(inputs)
             +" but got input of length "+str(len(inp)))
        feed_dict = {}
        for input_tensor, input_val in zip(inputs, inp):
            feed_dict[input_tensor] = input_val 
        sess = get_session()
        return sess.run(outputs, feed_dict=feed_dict)  
    return func_to_return

def get_gapped_kmer_embedding_func(filters, biases):
    '''
    Sourced from Avanti Shrikumar's MoDISCO repository: 
    https://github.com/kundajelab/tfmodisco/blob/d0827088f3718b477414a6042ec348e18d8ab09b/modisco/backend/tensorflow_backend.py#L81
    '''
    
    #filters should be: out_channels, rows, ACGT
    filters = filters.astype("float32").transpose((1,2,0))
    biases = biases.astype("float32")
    onehot_var = tf.placeholder(dtype=tf.float32,
                                shape=(None,None,None),
                                name="onehot")
    toembed_var = tf.placeholder(dtype=tf.float32,
                                 shape=(None,None,None),
                                 name="toembed")
    
    tf_filters = tf.convert_to_tensor(value=filters, name="filters")
    onehot_out = 1.0*(tf.cast(tf.greater(tf.nn.conv1d(
                    value=onehot_var,
                    filters=tf_filters,
                    stride=1,
                    padding='VALID') + biases[None,None,:], 0.0),
                    tf.float32))
    
    embedding_out = tf.reduce_sum(
                        input_tensor=(
                            tf.nn.conv1d(
                                value=toembed_var,
                                filters=tf_filters,
                                stride=1,
                                padding='VALID'))*(onehot_out),
        axis=1)
    func = compile_func(inputs=[onehot_var, toembed_var],
                        outputs=embedding_out)
    def batchwise_func(onehot, to_embed, batch_size, progress_update):
        return np.array(run_function_in_batches(
                            func=func,
                            input_data_list=[onehot, to_embed],
                            batch_size=batch_size,
                            progress_update=progress_update))
    return batchwise_func


def generate_gapped_kmers(kmer_len,
                          num_gaps,
                          alphabet_size=4,
                          num_mismatches=0):
    '''
    takes in a matrix of deepLIFT scores and/or fully connected layer embeddings and generates gapped kmer embedding 
    Sourced from Avanti Shrikumar's MoDISCO repository: 
    https://github.com/kundajelab/tfmodisco/blob/d0827088f3718b477414a6042ec348e18d8ab09b/modisco/affinitymat/core.py#L127
    '''
    nonzero_position_combos = list(itertools.combinations(
                        iterable=range(kmer_len),
                        r=(kmer_len-num_gaps)))
    
    letter_permutations = list(itertools.product(
                            *[list(range(alphabet_size)) for x in
                              range(kmer_len-num_gaps)]))
    filters = []
    biases = []
    
    unique_nonzero_positions = set()
    
    for nonzero_positions in nonzero_position_combos:
        string_representation = [" " for x in range(kmer_len)]
        
        for nonzero_position in nonzero_positions:
            string_representation[nonzero_position] = "X"
            
        nonzero_positions_string =("".join(string_representation)).lstrip().rstrip()
        
        if (nonzero_positions_string not in unique_nonzero_positions):
            unique_nonzero_positions.add(nonzero_positions_string) 
            for letter_permutation in letter_permutations:
                assert len(nonzero_positions)==len(letter_permutation)
                the_filter = np.zeros((kmer_len, alphabet_size)) 
                for nonzero_position, letter in zip(nonzero_positions, letter_permutation):
                    the_filter[nonzero_position, letter] = 1
                    
                filters.append(the_filter)
                biases.append(-(len(nonzero_positions)-1-num_mismatches))
                
    return np.array(filters), np.array(biases)

    
def core_compute_gapped_kmer_embedding(kmer_len,
                                       num_gaps,
                                       alphabet_size,
                                       imp_scores,
                                       rc=True,
                                       onehot=None,
                                       batch_size=100,
                                       progress_update=True):
    filters, biases=generate_gapped_kmers(kmer_len,
                                          num_gaps,
                                          alphabet_size=4)
    print("generated gapped kmers") 
    embed_func=get_gapped_kmer_embedding_func(filters,biases)
    print("got gapped kmer embedding function")
    if (onehot) is None:
        onehot = 1.0*(np.abs(impscores)>0)
    kmer_embeddings=embed_func(one_hot,impscores,batch_size=batch_size,progress_update=True)
    if (rc):
        kmer_embeddings_rev=embed_func(one_hot[:,::-1,::-1],impscores[:,::-1,::-1],batch_size=batch_size,progress_update=True)
        summed_kmer_embeddings=kmer_embeddings+kmer_embeddings_rev        
        return summed_kmer_embeddings
    else:
        return kmer_embeddings


def compute_gapped_kmer_embedding_wrapper(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    kmer_len=args.kmer_len
    num_gaps=args.num_gaps
    alphabet_size=args.alphabet_size
    imp_score_files=args.importance_score_files
    outf_files=args.outf
    batch_size=args.batch_size
    compute_gapped_kmer_embedding(kmer_len,num_gaps,alphabet_size,imp_score_files,outf_files,batch_size=batch_size,progress_update=True)
    
def compute_gapped_kmer_embedding(kmer_len,
                                  num_gaps,
                                  alphabet_size,
                                  imp_score_files,
                                  outf=None,
                                  batch_size=100,
                                  progress_update=True):
    filters, biases=generate_gapped_kmers(kmer_len,
                                          num_gaps,
                                          alphabet_size=4)
    print("generated gapped kmers") 
    embed_func=get_gapped_kmer_embedding_func(filters,biases)
    print("got gapped kmer embedding function")
    outputs=[] 
    for i in range(len(imp_score_files)):
        #get the current importance score file name 
        impscore_filename=imp_score_files[i]
        impscores = np.array([np.array( [[float(z) for z in y.split(",")]
                                for y in x.rstrip().split("\t")[2].split(";")])
                     for x in open(impscore_filename)])
        one_hot=1.0*(np.abs(impscores)>0)
        kmer_embeddings=embed_func(one_hot,impscores,batch_size=batch_size,progress_update=True)
        try:
            kmer_embeddings_rev=embed_func(one_hot[:,::-1,::-1],impscores[:,::-1,::-1],batch_size=batch_size,progress_update=True)
        except:
            pdb.set_trace() 
        summed_kmer_embeddings=kmer_embeddings+kmer_embeddings_rev        
        if outf is not None:
            cur_output_file=outf[i]
            print("writing gzip-compressed output file:"+cur_output_file)
            np.savez_compressed(cur_output_file,embeddings=summed_kmer_embeddings)
        else:
            outputs.append(summed_kmer_embeddings)
            
    #if no output file was provided, return the embeddings as numpy matrices. 
    if outf is None:
        return outputs 
    else:
        return 
    
def main():
    args=parse_args()
    compute_gapped_kmer_embedding_wrapper(args)
    
if __name__=="__main__":
    main()
    
