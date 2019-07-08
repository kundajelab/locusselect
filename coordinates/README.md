These coordinates were obtained by running the following code
in the PrepareData.ipynb notebook:

    #write bed list to file
    for idx,bedlist in enumerate(avocado_bed_list):
        outfile = open("coordinates_"+str(idx)+".bed", 'w')
        for coord in bedlist:
            chrom, start, end = coord
            chrom = chrom.decode("utf-8")
            start = start-1
            #when anna expands, it's flanksize, center, flank-1
            #so we should give her code a summit location of
            assert (end-start)%2 == 0
            flanksize = int((end-start)/2)
            outfile.write(chrom+"\t"+str(start)+"\t"+str(end)
                          +"\t.\t.\t.\t.\t.\t.\t"+str(flanksize)+"\n")
        outfile.close()
