import numpy as np 
def load_embedding(fname_npz):
    data=np.load(fname_npz,allow_pickle=True)
    regions=data['bed_entries']
    if 'embeddings' in data: 
        embeddings=data['embeddings']
        data_type='embedding'
    else:
        embeddings=data['deeplift']
        data_type='deeplift'
    return regions,embeddings,data_type
