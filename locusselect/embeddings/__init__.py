from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb 
#numpy & i/o
import warnings
import numpy as np
import argparse
import pysam
import pandas as pd

#import keras functions
import keras
from keras.engine.input_layer import Input 
from keras import callbacks as cbks
from keras.losses import *
from keras.models import Model 
from keras.layers import GlobalAveragePooling1D, AveragePooling1D, Flatten

#import dependencies from locusselect 
from locusselect.generators import *
from locusselect.custom_losses import *
from locusselect.metrics import recall, specificity, fpr, fnr, precision, f1
from locusselect.config import args_object_from_args_dict

def parse_args():
    parser=argparse.ArgumentParser(description='Provide a model yaml & weights files & a dataset, get model predictions and accuracy metrics')
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--max_queue_size",type=int,default=100)
    parser.add_argument('--model_hdf5',help='hdf5 file that stores the model')
    parser.add_argument('--weights',help='weights file for the model')
    parser.add_argument('--yaml',help='yaml file for the model')
    parser.add_argument('--json',help='json file for the model')
    parser.add_argument("--embedding_layer_number",type=int,default=None, help="model layer for which to calculate embedding")
    parser.add_argument("--input_layer_number",type=int,default=None) 
    parser.add_argument("--embedding_layer_name",type=str,default=None,help="name of embedding layer, alternative to embedding_layer_number")
    parser.add_argument("--input_layer_name",type=str,default=None)
    parser.add_argument('--input_bed_file',required=True,help='bed file with peaks to generate embedding')
    parser.add_argument('--batch_size',type=int,help='batch size to use to compute embeddings',default=1000)
    parser.add_argument('--ref_fasta')
    parser.add_argument('--flank',default=500,type=int)
    parser.add_argument('--center_on_summit',default=False,action='store_true',help="if this is set to true, the peak will be centered at the summit (must be last entry in bed file) and expanded args.flank to the left and right")
    parser.add_argument('--center_on_bed_interval',default=False,action='store_true')
    parser.add_argument("--output_npz_file",default=None,help="name of output file to store embeddings. The npz file will have fields \"bed_entries\" and \"embeddings\"")
    parser.add_argument("--expand_dims",default=False,action="store_true",help="set to True if using 2D convolutions, Fales if 1D convolutions (default)")
    parser.add_argument("--global_pool_on_position",default=False,action="store_true")
    parser.add_argument("--non_global_pool_on_position_size",type=int,default=None)
    parser.add_argument("--non_global_pool_on_position_stride",type=int,default=None) 
    parser.add_argument("--start_row",type=int,default=0)
    parser.add_argument("--num_rows",type=int,default=1000)
    return parser.parse_args()


def get_embeddings(args,model):
    data_generator=DataGenerator(args.input_bed_file,
                                 args.ref_fasta,
                                 batch_size=args.batch_size,
                                 center_on_summit=args.center_on_summit,
                                 center_on_bed_interval=args.center_on_bed_interval,
                                 flank=args.flank,
                                 expand_dims=args.expand_dims,
                                 return_indices=False)
    data_length=len(data_generator)*args.batch_size
    done=False
    start_row=0
    num_rows=args.num_rows
    embeddings=None
    while True:
        data_subgenerator=DataGenerator(args.input_bed_file,
                                        args.ref_fasta,
                                        batch_size=args.batch_size,
                                        center_on_summit=args.center_on_summit,
                                        center_on_bed_interval=args.center_on_bed_interval,
                                        flank=args.flank,
                                        expand_dims=args.expand_dims,
                                        start_row=start_row,
                                        num_rows=num_rows,
                                        return_indices=False)

        print("created data generator from {}".format(start_row))
        e=model.predict(data_subgenerator,
                        max_queue_size=args.max_queue_size,
                        workers=args.threads,
                        use_multiprocessing=True,
                        verbose=1)
        if(embeddings is None):
            embeddings = e
        else:
            embeddings = np.vstack((embeddings, e))
        start_row+=num_rows
        if start_row>=data_length:
            break

    print("got embeddings")
    bed_entries=data_generator.data_index
    print("got region labels")
    return np.asarray(bed_entries), embeddings
   
def get_embedding_layer_model(model,embedding_layer_number,input_layer_number, embedding_layer_name, input_layer_name):
    '''
    if input_seq_len is provided the model with use the central n base pairs of the input sequence as 
    input to the convolution stack 
    '''
    if input_layer_name is not None:
        assert embedding_layer_name is not None
        return Model(inputs=model.get_layer(input_layer_name).input,
                     outputs=model.get_layer(embedding_layer_name).output)
    else:
        assert input_layer_number is not None
        assert embedding_layer_number is not None 
        return Model(inputs=model.layers[input_layer_number].input,
                     outputs=model.layers[embedding_layer_number].output)

def add_positional_pooling(model,args):
    if args.embedding_layer_name is not None:
        target_layer=model.get_layer(args.embedding_layer_name)
    else: 
        target_layer=model._layers[args.embedding_layer_number]
    if (target_layer.__class__.__name__.startswith("Conv")==False):
        #We only want to change the input shape for a conv layer 
        return model
    
    #user did not specify pooling should be performed
    if (args.global_pool_on_position==False) and (args.non_global_pool_on_position_size is None):
        return model
    
    if args.global_pool_on_position==True:
        pooled_embedding = GlobalAveragePooling1D(data_format="channels_last")(model.output)
        flat_embedding=pooled_embedding
    elif args.non_global_pool_on_position_size is not None:
        if args.non_global_pool_on_position_stride is None:
            non_global_pool_on_position_stride=args.non_global_pool_on_position_size
        else:
            non_global_pool_on_position_stride=args.non_global_pool_on_position_stride
        pooled_embedding = AveragePooling1D(pool_size=args.non_global_pool_on_position_size, strides=args.non_global_pool_on_position_stride, padding="same", data_format="channels_last")(model.output)
        flat_embedding=Flatten()(pooled_embedding)
    #create graph of your new model
    new_model = Model(model.input,flat_embedding)
    return new_model

def reshape_model_inputs(model,new_input_shape,args):
    #we don't want to reshape the model if we are getting the embedding from a fully connected layer
    if args.embedding_layer_name is not None:
        target_layer=model.get_layer(args.embedding_layer_name)
    else: 
        target_layer=model._layers[args.embedding_layer_number]
    if (target_layer.__class__.__name__.startswith("Conv")==False):
        #We only want to change the input shape for a conv layer 
        return model
    if args.input_layer_name is not None:
        model.get_layer(args.input_layer_name)._batch_input_shape=new_input_shape
    elif args.input_layer_number is not None:
        model._layers[args.input_layer_number]._batch_input_shape=new_input_shape
    else:
        model._layers[0]._batch_input_shape = new_input_shape        
    new_model=keras.models.model_from_json(model.to_json())
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer:"+str(layer.name))
    return new_model

    

def get_model(args):
    from keras.utils.generic_utils import get_custom_objects
    custom_objects={"recall":recall,
                    "sensitivity":recall,
                    "specificity":specificity,
                    "fpr":fpr,
                    "fnr":fnr,
                    "precision":precision,
                    "f1":f1,
                    "ambig_binary_crossentropy":ambig_binary_crossentropy,
                    "ambig_mean_squared_error":ambig_mean_squared_error,
                    "MultichannelMultinomialNLL":MultichannelMultinomialNLL}
    get_custom_objects().update(custom_objects)
    if args.yaml!=None:
        from keras.models import model_from_yaml
        #load the model architecture from yaml
        yaml_string=open(args.yaml,'r').read()
        model=model_from_yaml(yaml_string)
        #load the model weights
        model.load_weights(args.weights)
        
    elif args.json!=None:
        from keras.models import model_from_json
        #load the model architecture from json
        json_string=open(args.json,'r').read()
        model=model_from_json(json_string)
        model.load_weights(args.weights)
        
    elif args.model_hdf5!=None: 
        #load from the hdf5
        from keras.models import load_model
        model=load_model(args.model_hdf5)
    print("got model architecture")
    print("loaded model weights")
    print(model.summary())
    
    return model


def compute_embeddings(args):
    if type(args)==type({}):
        args=args_object_from_args_dict(args) 
    
    #get the original model supplied by user
    model=get_model(args)
    print("loaded model")
    
    #reshape model inputs to take a variable size input
    if args.expand_dims is True:
        new_input_shape=(None,1,args.flank*2,4)
    else:
        new_input_shape=(None,args.flank*2,4)
        

    #get the model that returns embedding at user-specified layer
    embedding_layer_model=get_embedding_layer_model(model,args.embedding_layer_number, args.input_layer_number, args.embedding_layer_name, args.input_layer_name)
    print("obtained embedding layer model")

    embedding_layer_model=reshape_model_inputs(embedding_layer_model,new_input_shape,args)    
    
    #add a pooling layer to add positional invariance if the requested target embedding layer is a Convolution layer 
    pooled_flattened_model=add_positional_pooling(embedding_layer_model,args)

    print(pooled_flattened_model.summary())
    
    #get the embeddings of the input narrowPeak file peaks 
    bed_entries,embeddings=get_embeddings(args,pooled_flattened_model)
    if args.output_npz_file is not None:
        print("writing output file")
        np.savez_compressed(args.output_npz_file,bed_entries=bed_entries,embeddings=embeddings)
    return bed_entries,embeddings

    
def main():
    args=parse_args()
    compute_embeddings(args) 

    

if __name__=="__main__":
    main()
    
