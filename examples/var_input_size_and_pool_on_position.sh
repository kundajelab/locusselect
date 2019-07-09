#CUDA_VISIBLE_DEVICES=1 compute_embeddings \
#		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
#		    --weights conv.weights.hdf5 \
#		    --json conv.arch.json \
#		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
#		    --center_on_summit \
#		    --flank 125 \
#		    --output_npz_file k562_dnase_classification_embeddings.0.1.flank125.1.npz \
#		    --embedding_layer 1 \
#		    --expand_dims \
#		    --threads 40 \
#		    --global_pool_on_position
#

CUDA_VISIBLE_DEVICES=1 compute_embeddings \
		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
		    --weights conv.weights.hdf5 \
		    --json conv.arch.json \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 125 \
		    --output_npz_file tmp.npz \
		    --embedding_layer 1 \
		    --expand_dims \
		    --threads 40 \
		    --global_pool_on_position


