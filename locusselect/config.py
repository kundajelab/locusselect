import argparse
import pdb 
def args_object_from_args_dict(args_dict):
    #create an argparse.Namespace from the dictionary of inputs
    args_object=argparse.Namespace()
    #set the defaults for training/prediction/interpretation/cross-validation
    #training
    vars(args_object)['batch_size']=1000
    #prediction
    vars(args_object)['threads']=1
    vars(args_object)['max_queue_size']=100
    vars(args_ojbect)['tasks']=None
    vars(args_object)['batch_size']=1000
    vars(args_object)['flank']=500
    vars(args_ojbect)['center_on_summit']=False 
    for key in args_dict:
        vars(args_object)[key]=args_dict[key]
    args=args_object
    return args

    
    
