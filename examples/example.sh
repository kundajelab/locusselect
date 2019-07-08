#Paths are on the *brahma* server 
#classification model ("standard binned") (209865, 1000, 1) for layer -3 
CUDA_VISIBLE_DEVICES=1 compute_embeddings \
		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
		    --model_hdf5 /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/classification_init_dan_model/DNASE.K562.classification.SummitWithin200bpCenter.0 \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 500 \
		    --output_npz_file k562_dnase_classification_embeddings.0.-3.npz \
		    --embedding_layer -3 \
		    --expand_dims \
                   --threads 40 


#regression model ("standard binned") (209865, 1000, 1) for layer -2
CUDA_VISIBLE_DEVICES=1 compute_embeddings \
		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
		    --model_hdf5 /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/regression/DNASE.K562.regressionlabels.allbins.0 \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 500 \
		    --output_npz_file k562_dnase_regression_embeddings.0.-2.npz \
		    --embedding_layer -2 \
		    --expand_dims \
		    --threads 40 


#Profile model produces an embedding of dimension (209865, 3000, 32) for layer -2 
CUDA_VISIBLE_DEVICES=1 compute_embeddings \
		    --input_bed_file optimal_peak.narrowPeak.gz \
		    --weights 13kb_context_3b_prediction_dnase.hdf5 \
		    --json k562_dnase_profile_arch.json \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 6500 \
		    --output_npz_file k562_dnase_profile_embeddings_layer_-2.npz \
		    --embedding_layer -2 \
		    --threads 40

#Profile model produces an embedding of dimension (209865,3000,1) for layer -1 
CUDA_VISIBLE_DEVICES=1 compute_embeddings \
		    --input_bed_file optimal_peak.narrowPeak.gz \
		    --weights 13kb_context_3b_prediction_dnase.hdf5 \
		    --json k562_dnase_profile_arch.json \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 6500 \
		    --output_npz_file k562_dnase_profile_embeddings_layer_-1.npz \
		    --embedding_layer -1 \
		    --threads 40


# DeepLIFT scores for classification model
CUDA_VISIBLE_DEVICES=1 compute_deeplift_scores \
		    --input_bed_file /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/optimal_peak.narrowPeak.gz \
		    --model_hdf5 /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/classification_init_dan_model/DNASE.K562.classification.SummitWithin200bpCenter.0 \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 500 \
		    --output_npz_file k562_dnase_classification_0.deeplift.npz \
		    --deeplift_layer -2 \
		    --expand_dims \
		    --deeplift_reference shuffled_ref \
		    --deeplift_num_refs_per_seq 1 \
		    --batch_size 500 \
		    --task_index 0 \
		    --threads 20

# DeepLIFT scores for classification model
CUDA_VISIBLE_DEVICES=1 compute_deeplift_scores \
		    --input_bed_file optimal_peak.narrowPeak.gz \
		    --model_hdf5 /srv/scratch/annashch/deeplearning/encode4crispr/k562_dnase/regression/DNASE.K562.regressionlabels.allbins.0 \
		    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
		    --center_on_summit \
		    --flank 500 \
		    --output_npz_file k562_dnase_regression_0.deeplift.npz \
		    --deeplift_layer -1 \
		    --expand_dims \
		    --deeplift_reference shuffled_ref \
		    --deeplift_num_refs_per_seq 1 \
		    --batch_size 500 \
		    --task_index 0 \
		    --threads 20

