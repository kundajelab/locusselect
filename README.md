# locusselect

* extraction of data embeddings from deep learning model layers;

* computation of embedding distance for inputs;

* clustering and visualization of embeddings  with umap/tsne 

See examples/example.sh for usage examples. 

```
 compute_embeddings \
                    --input_bed_file optimal_peak.narrowPeak.gz \
                    --weights 13kb_context_3b_prediction_dnase.hdf5 \
                    --json k562_dnase_profile_arch.json \
                    --ref_fasta /mnt/data/annotations/by_release/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
                    --center_on_summit \
                    --flank 6500 \
                    --output_npz_file k562_dnase_profile_embeddings_layer_-2.npz \
                    --embedding_layer -2 \
                    --threads 40

```
