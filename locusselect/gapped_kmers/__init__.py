import itertools
import numpy as np
import tensorflow as tf 
import pdb

def get_gapped_kmer_embedding_func(filters, biases):
    '''
    Sourced from Avanti Shrikumar's MoDISCO repository: 
    https://github.com/kundajelab/tfmodisco/blob/d0827088f3718b477414a6042ec348e18d8ab09b/modisco/affinitymat/core.py#L127
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

def main():
    kmer_len=3
    num_gaps=1
    filters, biases=generate_gapped_kmers(kmer_len,
                                          num_gaps,
                                          alphabet_size=4)    
    pdb.set_trace() 
    embed_func=get_gapped_kmer_embedding_func(filters,biases)
    pdb.set_trace()
    
if __name__=="__main__":
    main()
    
