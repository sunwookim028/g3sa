#ifndef TB_KERNEL_H
#define TB_KERNEL_H

#define N_VALUE_T 'N'

#define GET_SUB_SCORE(subScore, query, target) \
	subScore = (query == target) ? _cudaMatchScore : -_cudaMismatchScore;\
	subScore = ((query == N_VALUE_T) || (target == N_VALUE_T)) ? 0 : subScore;\

#include "gasal_kernels.h"
#include "gasal_header.h"
#include "common.h"
/**
 * Called when DYNAMIC_TB is NOT defined.
 * It gets traceback directions from global memory that is calculated during forward phase.
 */

__global__ void traceback_kernel_old(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *global_direction, uint8_t *result_query, uint8_t *result_target, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length);
// uint64_t *global_direction_offsets) 
__global__ void traceback_kernel_cigar(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *global_direction, uint32_t *cigar, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, int* n_cigar, int bw);
//__global__ void traceback_kernel_dynamic(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t* cigar, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, short2 *dblock_row, short2 *dblock_col, int* n_cigar, int bw);
__global__ void traceback_kernel_square_grid(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t* cigar, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, short2 *dblock_row, short2 *dblock_col, int* n_cigar, int bw);
__global__ void traceback_kernel_square_grid_offset(uint8_t *unpacked_query_batch, uint8_t *unpacked_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t* cigar, gasal_res_t *device_res, int n_tasks, uint32_t maximum_sequence_length, short2 *dblock_row, short2 *dblock_col, uint64_t* tb_offset, int* n_cigar, int bw);


#endif