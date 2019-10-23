import pysam
import argparse 
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bed")
    parser.add_argument("--ref_fasta")
    parser.add_argument("--outf")
    return parser.parse_args()


def main():
    args=parse_args()
    outf=open(args.outf,'w')
    ref=pysam.FastaFile(args.ref_fasta)
    bed_regions=open(args.bed,'r').read().strip().split('\n')
    for region in bed_regions:
        region=region.split('\t')[0:3] 
        seq=ref.fetch(region[0],int(region[1]),int(region[2]))
        seq=seq.lower() 
        #calculate gc content
        g_count=seq.count('g')
        c_count=seq.count('c')
        t_count=seq.count('t')
        a_count=seq.count('a')
        gc_fraction=(g_count+c_count)/(g_count+c_count+a_count+t_count)
        outf.write(region[0]+'\t'+region[1]+'\t'+region[2]+'\t'+str(gc_fraction)+'\n')
    outf.close()

            
if __name__=="__main__":
    main()
    
