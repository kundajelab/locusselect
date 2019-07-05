#Paths are on the *brahma* server 
#classification model ("standard binned") 
#CUDA_VISIBLE_DEVICES=1 compute_embeddings \
#		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
#		    --model_hdf5 /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/classification_init_dan_model/DNASE.K562.classification.SummitWithin200bpCenter.0 \
#		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
#		    --center_on_summit \
#		    --flank 500 \
#		    --output_hdf5 k562_dnase_classification_embeddings.0.hdf5 \
#		    --embedding_layer -2


#Profile model 
CUDA_VISIBLE_DEVICES=1 compute_embeddings \
		    --input_bed_file optimal_peak.narrowPeak.gz \
		    --weights 13kb_context_3b_prediction_dnase.hdf5 \
		    --json k562_dnase_profile_arch.json \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 6500 \
		    --output_hdf5 k562_dnase_profile_embeddings.hdf5 \
		    --embedding_layer -2 \
		    --threads 40
