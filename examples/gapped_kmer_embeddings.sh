CUDA_VISIBLE_DEVICES=3 compute_kmer_embeddings \
		    --importance_score_files coordinates_0.K562_m1_r1.model.explanation.txt \
		    --kmer_len 6 \
		    --num_gaps 1 \
		    --alphabet_size 4 \
		    --batch_size 100 \
		    --outf gkmexplain.coord.embeddings.BCL11A.npz
