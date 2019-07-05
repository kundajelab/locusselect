#Paths are on the *brahma* server 
#classification model ("standard binned") 
#CUDA_VISIBLE_DEVICES=1 compute_embeddings \
#		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
#		    --model_hdf5 /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/classification_init_dan_model/DNASE.K562.classification.SummitWithin200bpCenter.0 \
#		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
#		    --center_on_summit \
#		    --flank 500 \
#		    --output_npz_file k562_dnase_classification_embeddings.0.npz \
#		    --embedding_layer -2


#Profile model produces an embedding of dimension (209865, 3000, 32) for layer -2 
#CUDA_VISIBLE_DEVICES=1 compute_embeddings \
#		    --input_bed_file optimal_peak.narrowPeak.gz \
#		    --weights 13kb_context_3b_prediction_dnase.hdf5 \
#		    --json k562_dnase_profile_arch.json \
#		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
#		    --center_on_summit \
#		    --flank 6500 \
#		    --output_npz_file k562_dnase_profile_embeddings_layer_-2.npz \
#		    --embedding_layer -2 \
#		    --threads 40

#Profile model produces an embedding of dimension for layer -1 
CUDA_VISIBLE_DEVICES=2 compute_embeddings \
		    --input_bed_file optimal_peak.narrowPeak.gz \
		    --weights 13kb_context_3b_prediction_dnase.hdf5 \
		    --json k562_dnase_profile_arch.json \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 6500 \
		    --output_npz_file k562_dnase_profile_embeddings_layer_-1.npz \
		    --embedding_layer -1 \
		    --threads 40
