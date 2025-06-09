#ifndef CHAINING_KERNEL_H
#define CHAINING_KERNEL_H

#include "common.h"
#include "minimap.h"
#include "mmpriv.h"
#include "datatype.h"


typedef struct {
	int r_ofs; // offset of its parent read - used to access query sequence & anchor data
    int a_ofs;
	int r_n; // temp - for debugging
	mm_reg1_t r; // chaining metadata
	int qlen; // length of original query read
	bool dropped;
	int n_a;
	uint8_t flag; // extension stage flag 
    int gap_ofs;
    int n_gap;
} mm_chain_t;

#ifdef __CUDACC__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>


__global__
void range_inexact_segment_block(uint64_t* x, int* a_len, int2* r, bool* pass, int max_dist_x, int n_task, int32_t* sc, int64_t* p);
__global__
void range_inexact(uint64_t* x, int* offset, int* a_len, uint2* r, int* n_r, int max_dist_x, int n_task);
__global__ 
void range_exact_naive(uint64_t* x, uint64_t* y, int* offset, int* a_len, uint2* r, int* n_r, int max_dist_x, int n_task);
__global__ 
void chain_naive_exact_range(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const uint2* range, 
                    int* offset, int* r_len, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task);
__global__ 
void chain_naive_short(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const uint2* range, 
                    int* offset, int* r_len, int* a_len, bool* pass, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task, int* long_num, 
                    int* long_buf_st, int* long_buf_len, int* mid_num, int* mid_buf_st, int* mid_buf_len);

__global__ 
void chain_segmented_short(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const uint2* range, 
                    int* offset, int* r_len, int* a_len, bool* pass, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task, int* long_num, 
                    int* long_buf_st, int* long_buf_len, int* mid_num, int* mid_buf_st, int* mid_buf_len);

__global__ 
void chain_naive_inexact_range(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const uint2* range, 
                    int* offset, int* r_len, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task, int* long_num, 
                    int* long_buf_st, int* long_buf_len, int* mid_num, int* mid_buf_st, int* mid_buf_len);
 
__global__ 
void chain_naive_long(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const int* range_st, const int* range_len, 
                    int* offset, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task);

__global__ 
void chain_tiled_long(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const int* range_st, const int* range_len, 
                    int* offset, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task);

__global__ 
void chain_tiled_reg_long(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const int* range_st, const int* range_len, 
                    int* offset, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task);
 
/* Bactracking kernel */

__global__
void chain_backtrack(int* n_a, uint64_t* g_ax, uint64_t* g_ay, int32_t* g_sc, int64_t *g_p, uint64_t* g_u, 
                    int64_t* g_zx, int64_t* g_zy, int32_t* g_t, int64_t* g_v,
                    int* offset, int* r_offset, int32_t min_cnt, int32_t min_sc, int32_t max_drop,
                    uint32_t hash, int* d_len, int is_qstrand, int* n_chain, mm_chain_t* g_c, int n_task, int* g_n_v, bool* dropped, bool final);

__global__ void mm_set_chain(int* g_na, int n_task,  uint64_t* g_ax, uint64_t* g_ay, int* offset, int* ofs_end, int32_t* g_sc, int64_t *g_p, uint64_t* g_u, 
                        int64_t* g_zx, int64_t* g_zy, int32_t* g_t, int64_t* g_v);

__global__ void mm_chain_backtrack(int* n_a, uint64_t* g_ax, uint64_t* g_ay, int32_t* g_sc, int64_t *g_p, uint64_t* g_u, 
                            int64_t* g_zx, int64_t* g_zy, int32_t* g_t, int64_t* g_v,
                            int* offset, int32_t min_cnt, int32_t min_sc, int32_t max_drop,
                            int n_task, int* g_n_v, int* g_n_u, int* g_n_z, int* ofs_end);

__global__ void mm_chain_backtrack_parallel(int* n_a, uint64_t* g_ax, uint64_t* g_ay, int32_t* g_sc, int64_t *g_p, uint64_t* g_u, 
                    int64_t* g_zx, int64_t* g_zy, int32_t* g_t, int64_t* g_v,
                    int* offset, int32_t min_cnt, int32_t min_sc, int32_t max_drop,
                    int n_task, int* g_n_v, int* g_n_z, int* ofs_end);


__global__ void mm_filter_anchors(int* n_a, int* offset, int32_t min_sc, int32_t* g_sc, int64_t* g_zx, int64_t* g_zy, 
                                int* ofs_out, int32_t* g_t,  int* g_n_z, int* g_n_v, int n_task);

__global__ void mm_gen_regs(int* n_a, uint64_t* g_ax, uint64_t* g_ay, uint64_t* g_u, 
                    int64_t* g_zx, int64_t* g_zy, int64_t* g_v, int* g_n_v, int* ofs_end, uint32_t hash,
                    int* offset, int* r_offset, int* d_len, int is_qstrand, int* n_chain, mm_chain_t* g_c, int n_task, bool* dropped);

__global__ void mm_gen_regs_dp(int* n_a, uint64_t* g_ax, uint64_t* g_ay, uint64_t* g_u, 
                        int64_t* g_zx, int64_t* g_zy, int64_t* g_v, int* g_n_v, int* g_n_u, int* ofs_end, uint32_t hash,
                        int* offset, int* r_offset, int* d_len, int is_qstrand, int* n_chain, mm_chain_t* g_c, int n_task, bool* dropped);

__global__ void mm_gen_regs1(int* n_a, uint64_t* g_ax, uint64_t* g_ay, uint64_t* g_u, 
                    int64_t* g_zx, int64_t* g_zy, int64_t* g_v, int* g_n_v, int* ofs_end, uint32_t hash,
                    int* offset, int* r_offset, int* d_len, int is_qstrand, int* n_chain, mm_chain_t* g_c, int n_task, bool* dropped);

__global__ void mm_gen_regs2(int* n_a, uint64_t* g_ax, uint64_t* g_ay, uint64_t* g_u, 
                        int64_t* g_zx, int64_t* g_zy, int64_t* g_v, int* g_n_v, int* g_n_u, int* ofs_end, uint32_t hash,
                        int* offset, int* r_offset, int* d_len, int is_qstrand, int* n_chain, mm_chain_t* g_c, int n_task, bool* dropped);
__global__
void chain_post( // mm_mapopt_t *opt, 
            int max_chain_gap_ref, int* qlen, int *n_regs, int* c_ofs, int* a_ofs, mm_chain_t *g_c, uint64_t *ax, uint64_t* ay, bool* g_dropped, int* g_w, uint64_t* g_cov, int n_task);

/* RMQ chaining kernel */
__global__
void range_inexact_rmq(uint64_t* x, int* offset, int* a_len, uint2* r, int* n_r, int max_dist_x, int n_task);

__global__ 
void chain_naive_inexact_range_rmq(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const uint2* range, 
                    int* offset, int* r_len, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task, int* long_num, int* long_buf_st, int* long_buf_len, int* mid_num, int* mid_buf_st, int* mid_buf_len);
                    
__global__ 
void chain_segmented_short_rmq(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const uint2* range, 
                                        int* offset, int* r_len, int* a_len, bool* pass, 
                                        int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task, int* long_num, int* long_buf_st, int* long_buf_len, int* mid_num, int* mid_buf_st, int* mid_buf_len);
                     
__global__ 
void chain_tiled_reg_long_rmq(int32_t* sc, int64_t* p, const uint64_t* anchors_x, const uint64_t* anchors_y, const int* range_st, const int* range_len, 
                    int* offset, int* a_len, 
                    int max_dist_x, int max_dist_y, int bw, int is_cdna, float chn_pen_gap, float chn_pen_skip, int n_task);

#endif // __CUDACC__
#endif // SEEDING_KERNEL_H
