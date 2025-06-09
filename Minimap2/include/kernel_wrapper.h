#include <stdint.h>
#include <vector>
#include "common.h"
#include "minimap.h"
#include "mmpriv.h"
#include "khash.h"
#include "krmq_cuda.h"
#include "seeding_kernel.h"
//#include "chaining_kernel.h"
#include "device_mem.h"


typedef struct {
	int32_t rid, rev; // TODO: rev doesn't need 32 bits 
	int32_t rs, rs0, rs1;
	int32_t re, re0, re1;
	int32_t qs, qs0, qs1;
	int32_t qe, qe0, qe1;
} mm_pos_t;

// typedef struct {
// 	int r_ofs; // offset of its parent read - used to access query sequence & anchor data
// 	int r_n; // temp - for debugging
// 	mm_reg1_t r; // chaining metadata
// 	int qlen; // length of original query read
// 	bool dropped;
// 	int n_a;
// 	uint8_t flag; // extension stage flag 
// } mm_chain_t;


typedef struct{
	int len;
	int ofs;
} batch_info_t; 

void gpu_kernel_wrapper(std::vector<char> &seqs, std::vector<int> &lens, std::vector<int> &ofs,
	int* rid, int len, int w, int k, int32_t* h_sc, int64_t* h_p, mm_idx_t* mi, std::vector<mm128_t> &minimizer, std::vector<int> &minimizer_n, int *n_a, int* ofs_a ,uint64_t* h_x, uint64_t* h_y,
	std::vector<uint64_t> cpu_x, std::vector<uint64_t> cpu_y, std::vector<int> a_num, std::vector<int> &anchor_offset,
	mm_chain_t* results, uint32_t* h_cigar, int* h_n_cigar, int* n_alignment,
	int* cpu_n_v, int* cpu_n_u, mm_chain_t* cpu_chain, int* cpu_n_chain, int tot_anchors);
							
mm_idx_t* transfer_index(mm_idx_t* mi);

mm_idx_t* transfer_index_multigpu(mm_idx_t* h_idx);

void* fetch_memory_mempool(void** memPoolPtr, int length, int typeSize, uint64_t* ptrIndex);

void seeding_kernel_wrapper(char* seqs, int* lens, int* ofs, int* rid, mm128_t* mmi, int* n_mmi, int batch_size, int w, int k,
                            mm_idx_t* idx, uint64_t* ax, uint64_t* ay, int* n_seed, int* anchor_ofs, int* tmp_n, uint64_t** tmp_cr, mm_seed_t* tmp_seed,int32_t* num_anchors);
							
void gpu_initialize(int gpu_id, mm_idx_t* mi, device_pointer_batch_t* device_ptrs, uint64_t max_seq_len, int batch_size);

							
//void gpu_mm2_kernel_wrapper(char* h_seqs, int* h_lens, int* h_ofs, device_pointer_batch_t* device_ptrs, int w, int k, uint64_t max_seq_len, int batch_size);

void gpu_mm2_kernel_wrapper(//int* rid, 
    char* h_seqs, int* h_lens, int* h_ofs, device_pointer_batch_t* device_ptrs, int w, int k, uint64_t max_seq_len, int batch_size,
    // debugging purpose : remove later
    int *n_a, int* ofs_a ,uint64_t* h_x, uint64_t* h_y, // seed test
    int* cpu_n_v, int* cpu_n_u, mm_chain_t* cpu_chain, int* cpu_n_chain, // post chain test
    int32_t* h_sc, int64_t* h_p, // chain test
    uint32_t* cpu_qlens, uint32_t* cpu_rlens, int8_t* cpu_qseqs, int8_t* cpu_rseqs, // extend test
    uint32_t* h_cigar, int* h_n_cigar, int* h_score, // output data
    uint8_t* h_dropped // fallback reads
);


#ifdef __CUDACC__

void chaining_kernel_wrapper_dp(uint64_t* ax, uint64_t* ay, int* ofs, int* n_a, uint2* range, int* n_range, int32_t* sc, int64_t* p, bool* seg_pass, int seg_num, int batch_size,
                            int* long_range_buf_st, int* long_range_buf_len, int* mid_range_buf_st, int* mid_range_buf_len);
							
void chaining_kernel_wrapper_rmq(uint64_t* ax, uint64_t* ay, int* ofs, int* n_a, uint2* range, int* n_range, int32_t* sc, int64_t* p, bool* seg_pass, int seg_num, int batch_size,
                            int* long_range_buf_st, int* long_range_buf_len, int* mid_range_buf_st, int* mid_range_buf_len);

#endif // __CUDACC__