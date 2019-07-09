#!/usr/bin/env python
import numpy as np

negsets = ['r1', 'r2']
possets = ['m1', 'm2', 'm3', 'm4']

for coordidx in [0,1,2,3,4,5]: 
    print("On coordset "+str(coordidx))
    files_to_average = ["coordinates_"+str(coordidx)+"/K562_"+posset+"_"+negset+".explanation.txt"
                        for posset in possets
                        for negset in negsets]
    coords = [] 
    preds = []
    explanations = []
    for fileidx,filename in enumerate(files_to_average):
        print("on filename",filename)
        for lineidx,line in enumerate(open(filename)):
            coord,pred,explanation = line.rstrip().split("\t")
            pred = float(pred)
            explanation = np.array([[float(z) for z in y.split(",")]
                                   for y in explanation.split(";")]) 
            if (fileidx==0):
                coords.append(coord) 
                preds.append(float(pred))
                explanations.append(explanation) 
            else:
                assert coords[lineidx] == coord
                preds[lineidx] += pred 
                explanations[lineidx] += explanation
    preds = np.array(preds)/len(files_to_average)
    explanations = np.array(explanations)/len(files_to_average)
    
    outfile = open("coordinates_"+str(coordidx)+"/averaged_"+str(len(files_to_average))+".explanation.txt", 'w')
    for coord,pred,explanation in zip(coords,preds,explanations):
        outfile.write(coord+"\t"+str(pred)+"\t"+(";".join([",".join([str(basescore) for basescore in posscores])
                                                     for posscores in explanation]))+"\n") 
    outfile.close() 
