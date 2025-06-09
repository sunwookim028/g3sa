#include "agatha_kernel.h"
#include "kernel_wrapper.h"

#define AGATHA_WARP_NUM 4
#define AGATHA_THREAD_NUM 256
#define SHARED_SIZE AGATHA_THREAD_NUM / AGATHA_WARP_NUM

__global__ void agatha_sort(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID

	uint32_t query_len, ref_len, packed_query_len, packed_ref_len;

	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	if (tid < n_tasks) {

		query_len = query_batch_lens[tid];
		ref_len = target_batch_lens[tid];
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);

		global_ub_idx[tid] = make_short2((packed_ref_len + packed_query_len-1), static_cast<int16_t>(tid));
	}
	
	return;


}


__global__ void agatha_kernel(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint4 *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{
    /*Initial kernel setup*/

	int bw = BW_EXTEND;
	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;
	int test_id;

	// Initializing max score and its idx
    int32_t max_score = 0; 
	int32_t max_ref_idx = 0; 
    int32_t prev_max_score = 0;
    int32_t max_query_idx = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* antidiag_max = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
		
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// check if alignment in the chain is done 


		//if(warp_id==0) printf("u_id: %d | ofs: r(%d), q(%d), len: r(%d), q(%d)\n", ub_idx, packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len);
		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= bw? k: MINUS_INF2; 	
			}
		}
		// fill shared memory with initial value
		for (m = 0; m < total_shm; m++) {
			antidiag_max[real_warp_id + m*32] = INT_MIN;
		}

		__syncwarp();

		// Initialize variables
		max_score = 0; 
		prev_max_score = 0;
		max_ref_idx = 0; 
    	max_query_idx = 0;
		terminated = false;

		// check termination condition

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - bw)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + bw)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];

					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - bw)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + bw)) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;
						
						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_BOUNDARY();
								}
							} else {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE();
								}
							}
							
							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							query_idx++;

						}

					}
					

					packed_query_idx++;
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;

			/* Termination Condition & Score Update */
			if (!terminated) {
				for (diag_idx = i<<3; diag_idx < last_diag; diag_idx++) {
					if (diag_idx <prev_max_score) {
						m = diag_idx&(total_shm-1);
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = diag_idx-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (diag_idx-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (diag_idx-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								// write zdrop information to global drop array
								terminated = true;
								break;
							}
						}
						// reset shared memory buffer for next slice
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
			}
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = max_score;
			else if (warp_id==2) shared_job[24+(warp_num&3)] = (max_ref_idx<<16) + max_query_idx;
 
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// In the case of (2), check the termination condition & score update for the last diagonal block
				if (!terminated) {
					diag_idx = (i*packed_len)&(total_shm-1);
					for (k = i*packed_len, m = diag_idx; m < diag_idx+packed_len; m++, k++) {
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = k-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (k-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (k-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					device_res->aln_score[ub_idx] = max_score;//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[ub_idx] = max_query_idx;//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[ub_idx] = max_ref_idx;//copy the end position on target_batch sequence to the output array in the GPU mem
					//if(max_score > 0)
					// printf("[idx: %d] score: %d, q_end: %d, t_end: %d, qlen: %d, rlen: %d\n", ub_idx,max_score, max_query_idx, max_ref_idx, query_len, ref_len);
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						max_score = shared_job[20+m];
						max_ref_idx = shared_job[24+m];
						max_query_idx = max_ref_idx&65535;
						max_ref_idx = max_ref_idx>>16;
						
						// reset the flag
						terminated = false;

						// reset shared memory buffer
						for (m = 0; m < total_shm; m++) {
							antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
						}
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}
		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}

__global__ void agatha_kernel_static_tb(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint32_t *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{
    /*Initial kernel setup*/

	int bw = BW_EXTEND;
	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;
	int test_id;

	// Initializing max score and its idx
    int32_t max_score = 0; 
	int32_t max_ref_idx = 0; 
    int32_t prev_max_score = 0;
    int32_t max_query_idx = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* antidiag_max = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
		
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// check if alignment in the chain is done 
	
		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= bw? k: MINUS_INF2; 	
			}
		}
		// fill shared memory with initial value
		for (m = 0; m < total_shm; m++) {
			antidiag_max[real_warp_id + m*32] = INT_MIN;
		}

		__syncwarp();

		// Initialize variables
		max_score = 0; 
		prev_max_score = 0;
		max_ref_idx = 0; 
    	max_query_idx = 0;
		terminated = false;

		// check termination condition

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - bw)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + bw)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];

					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - bw)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + bw)) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;
						
						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;
							
							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_BOUNDARY_TB(packed_tb_matrices[ub_idx*max_query_len*max_query_len/8 + packed_ref_idx * max_query_len + query_idx]);
								}
							} else {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_TB(packed_tb_matrices[ub_idx*max_query_len*max_query_len/8 + packed_ref_idx * max_query_len + query_idx]);
								}
							}
							
							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							query_idx++;

						}

					}
					

					packed_query_idx++;
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;

			/* Termination Condition & Score Update */
			if (!terminated) {
				for (diag_idx = i<<3; diag_idx < last_diag; diag_idx++) {
					if (diag_idx <prev_max_score) {
						m = diag_idx&(total_shm-1);
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = diag_idx-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (diag_idx-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (diag_idx-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								// write zdrop information to global drop array
								terminated = true;
								break;
							}
						}
						// reset shared memory buffer for next slice
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
			}
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = max_score;
			else if (warp_id==2) shared_job[24+(warp_num&3)] = (max_ref_idx<<16) + max_query_idx;
 
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// In the case of (2), check the termination condition & score update for the last diagonal block
				if (!terminated) {
					diag_idx = (i*packed_len)&(total_shm-1);
					for (k = i*packed_len, m = diag_idx; m < diag_idx+packed_len; m++, k++) {
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = k-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (k-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (k-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					device_res->aln_score[ub_idx] = max_score;//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[ub_idx] = max_query_idx;//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[ub_idx] = max_ref_idx;//copy the end position on target_batch sequence to the output array in the GPU mem
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						max_score = shared_job[20+m];
						max_ref_idx = shared_job[24+m];
						max_query_idx = max_ref_idx&65535;
						max_ref_idx = max_ref_idx>>16;
						
						// reset the flag
						terminated = false;

						// reset shared memory buffer
						for (m = 0; m < total_shm; m++) {
							antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
						}
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}
		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}

__global__ void agatha_kernel_dynamic_tb(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, short2 *dblock_row, short2 *dblock_col, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top)
{
    /*Initial kernel setup*/

	int bw = BW_EXTEND;
	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;
	int test_id;

	// Initializing max score and its idx
    int32_t max_score = 0; 
	int32_t max_ref_idx = 0; 
    int32_t prev_max_score = 0;
    int32_t max_query_idx = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* antidiag_max = (int32_t*)(shared_maxHH+(threadIdx.x/32)*total_shm*32);
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
		
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// check if alignment in the chain is done 

		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= bw? k: MINUS_INF2; 	
			}
		}
		// fill shared memory with initial value
		for (m = 0; m < total_shm; m++) {
			antidiag_max[real_warp_id + m*32] = INT_MIN;
		}

		__syncwarp();

		// Initialize variables
		max_score = 0; 
		prev_max_score = 0;
		max_ref_idx = 0; 
    	max_query_idx = 0;
		terminated = false;

		// check termination condition

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - bw)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + bw)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];

					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - bw)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + bw)) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;
						
						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_BOUNDARY();
								}
							} else {
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE();
								}
							}
							
							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							//--------------- save dblock row -------------
							if ((ref_idx+m-1) % DBLOCK_SIZE == 0) {
								dblock_row[ub_idx * max_query_len * max_query_len / DBLOCK_SIZE + max_query_len * (ref_idx+m-1) / DBLOCK_SIZE + query_idx] = HD;
							}
							//---------------------------------------------

							query_idx++;

						}

					}
					

					packed_query_idx++;

					//-------------- save dblock col -------------------
					if ((packed_query_idx<<3)%DBLOCK_SIZE == 0 && active) { // TODO: merge with the loop below? idk
						short2 tmp_HD;
						for (int ridx = 0; ridx < 8; ridx++) {
							tmp_HD.x = h[ridx+1];
							tmp_HD.y = f[ridx+1];
							dblock_col[ub_idx * max_query_len * max_query_len / DBLOCK_SIZE 
							+ max_query_len * (packed_query_idx<<3) / DBLOCK_SIZE  
							+ ref_idx + ridx] = tmp_HD;
						}
					}
					//--------------------------------------------------
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;

			/* Termination Condition & Score Update */
			if (!terminated) {
				for (diag_idx = i<<3; diag_idx < last_diag; diag_idx++) {
					if (diag_idx <prev_max_score) {
						m = diag_idx&(total_shm-1);
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = diag_idx-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (diag_idx-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (diag_idx-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								// write zdrop information to global drop array
								terminated = true;
								break;
							}
						}
						// reset shared memory buffer for next slice
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
			}
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = max_score;
			else if (warp_id==2) shared_job[24+(warp_num&3)] = (max_ref_idx<<16) + max_query_idx;
 
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// In the case of (2), check the termination condition & score update for the last diagonal block
				if (!terminated) {
					diag_idx = (i*packed_len)&(total_shm-1);
					for (k = i*packed_len, m = diag_idx; m < diag_idx+packed_len; m++, k++) {
						temp = __reduce_max_sync(same_threads, antidiag_max[(m<<5)+real_warp_id]);
						if ((temp>>16) > max_score) {				
							max_score = temp>>16;
							max_ref_idx = (temp&65535);
							max_query_idx = k-max_ref_idx; 
						} else if ( (temp&65535) >= max_ref_idx && (k-(temp&65535)) >= max_query_idx) {
							int tl =  (temp&65535) - max_ref_idx, ql = (k-(temp&65535)) - max_query_idx, l;
							l = tl > ql? tl - ql : ql - tl;
							if (_cudaZThreshold >= 0 && max_score - (temp>>16) > _cudaZThreshold + l*_cudaGapExtend) {
								// Termination condition is met
								terminated = true;
								break;
							}
						}
						antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
					}
				}
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					device_res->aln_score[ub_idx] = max_score;//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[ub_idx] = max_query_idx;//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[ub_idx] = max_ref_idx;//copy the end position on target_batch sequence to the output array in the GPU mem
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						max_score = shared_job[20+m];
						max_ref_idx = shared_job[24+m];
						max_query_idx = max_ref_idx&65535;
						max_ref_idx = max_ref_idx>>16;
						
						// reset the flag
						terminated = false;

						// reset shared memory buffer
						for (m = 0; m < total_shm; m++) {
							antidiag_max[(m<<5)+real_warp_id]=INT_MIN;
						}
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}
		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}


__global__ void agatha_kernel_approx_static_tb(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, uint32_t *packed_tb_matrices, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top,
								bool* dropped, int bw)
{
    /*Initial kernel setup*/

	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;

	int test_id = 332;

	// Initializing max score and its idx
	__shared__ int32_t global_max_score[SHARED_SIZE]; 
	__shared__ int32_t global_max_ref_idx[SHARED_SIZE]; 
	__shared__ int32_t global_max_query_idx[SHARED_SIZE]; 
    __shared__ int32_t max_score[SHARED_SIZE]; 
	__shared__ int32_t max_ref_idx[SHARED_SIZE]; 
	__shared__ int32_t max_query_idx[SHARED_SIZE];
	bool max_block = true;

    int32_t prev_max_score = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	int warp_num_block = threadIdx.x / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
		
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= bw? k: MINUS_INF2; 	
			}
		}

		// Initialize variables
		global_max_score[warp_num_block] = 0;
		global_max_ref_idx[warp_num_block] = 0;
		global_max_query_idx[warp_num_block] = 0;
		max_score[warp_num_block] = 0; 
		prev_max_score = 0;
		max_ref_idx[warp_num_block] = 0; 
    	max_query_idx[warp_num_block] = 0;
		// terminated = false;

		// check termination condition
		terminated = dropped[ub_idx]; // TODO: fix this to be per-warp variable 

		__syncwarp();

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - bw)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + bw)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];
			
					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - bw)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + bw)) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;
						
						// set max index value if this thread has max cell in its 8*8 block
						if(((max_ref_idx[warp_num_block] >= ref_idx-1) && (max_ref_idx[warp_num_block] <= ref_idx+7)) || ((max_query_idx[warp_num_block] >= query_idx-1) && (max_query_idx[warp_num_block] <= query_idx+7)))
							max_block = true;
						else max_block = false;

						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								if(max_block){
									#pragma unroll 8
									for (l = 28, m = 1; m < 9; l -= 4, m++) {
										CORE_COMPUTE_BOUNDARY_APPROX_MAX_TB(packed_tb_matrices[ub_idx*max_query_len*max_query_len/8 + packed_ref_idx * max_query_len + query_idx]);
									}
								}
								else{
									#pragma unroll 8
									for (l = 28, m = 1; m < 9; l -= 4, m++) {
										CORE_COMPUTE_BOUNDARY_APPROX_TB(packed_tb_matrices[ub_idx*max_query_len*max_query_len/8 + packed_ref_idx * max_query_len + query_idx]);
	
									}
								}
							} else if(max_block){
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_APPROX_MAX_TB(packed_tb_matrices[ub_idx*max_query_len*max_query_len/8 + packed_ref_idx * max_query_len + query_idx]);
								}
							} else{
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_APPROX_TB(packed_tb_matrices[ub_idx*max_query_len*max_query_len/8 + packed_ref_idx * max_query_len + query_idx]);
								}
							}
							

							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							query_idx++;

						}

					}
					

					packed_query_idx++;
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;

			// check global max and z-drop condition for this slice
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = warp_num_block;
			
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					device_res->aln_score[ub_idx] = max_score[warp_num_block];//copy the max score to the output array in the GPU mem
					device_res->query_batch_end[ub_idx] = max_query_idx[warp_num_block];//copy the end position on query_batch sequence to the output array in the GPU mem
					device_res->target_batch_end[ub_idx] = max_ref_idx[warp_num_block];//copy the end position on target_batch sequence to the output array in the GPU mem
					dropped[ub_idx] = terminated;
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						warp_num_block = shared_job[20+m];
						
						// reset the flag
						terminated = dropped[ub_idx];
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}
		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();



	}
	
	return;


}

__global__ void agatha_kernel_approx_dynamic_tb(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, short2 *dblock_row, short2 *dblock_col, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top,
								bool* dropped, int bw)
{
    /*Initial kernel setup*/

	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;

	// Initializing max score and its idx
	__shared__ int32_t global_max_score[SHARED_SIZE]; 
	__shared__ int32_t global_max_ref_idx[SHARED_SIZE]; 
	__shared__ int32_t global_max_query_idx[SHARED_SIZE]; 
    __shared__ int32_t max_score[SHARED_SIZE]; 
	__shared__ int32_t max_ref_idx[SHARED_SIZE]; 
	__shared__ int32_t max_query_idx[SHARED_SIZE];
	bool max_block = true;

    int32_t prev_max_score = 0;

	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses
	
	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	int warp_num_block = threadIdx.x / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {
		
		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? global_ub_idx[n_tasks-(job_idx>>2)-1].y: global_ub_idx[job_idx-(job_idx>>2)-1].y;
		
		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// check if alignment in the chain is done 
	
		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= bw? k: MINUS_INF2; 	
			}
		}

		// Initialize variables
		global_max_score[warp_num_block] = 0;
		global_max_ref_idx[warp_num_block] = 0;
		global_max_query_idx[warp_num_block] = 0;
		max_score[warp_num_block] = 0; 
		prev_max_score = 0;
		max_ref_idx[warp_num_block] = 0; 
    	max_query_idx[warp_num_block] = 0;
		// terminated = false;

		// check termination condition
		terminated = dropped[ub_idx]; // TODO: fix this to be per-warp variable 

		__syncwarp();

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk

		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			
			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - bw)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + bw)/2/packed_len);
			finished_blocks = slice_start;
			
			if (slice_start > slice_end) {
				terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not
				
				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];
			
					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}
						
					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - bw)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + bw)) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}
					
				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;

					
						// set max index value if this thread has max cell in its 8*8 block
						if(((max_ref_idx[warp_num_block] >= ref_idx-1) && (max_ref_idx[warp_num_block] <= ref_idx+7)) || ((max_query_idx[warp_num_block] >= query_idx-1) && (max_query_idx[warp_num_block] <= query_idx+7)))
							max_block = true;
						else max_block = false;

						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence
							
							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								if(max_block){
									#pragma unroll 8
									for (l = 28, m = 1; m < 9; l -= 4, m++) {
										CORE_COMPUTE_BOUNDARY_APPROX_MAX();
									}
								}
								else{
									#pragma unroll 8
									for (l = 28, m = 1; m < 9; l -= 4, m++) {
										CORE_COMPUTE_BOUNDARY_APPROX();
	
									}
								}
							} else if(max_block){
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_APPROX_MAX();
								}
							} else{
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_APPROX();
								}
							}
							
							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							//--------------- save dblock row -------------
							if ((ref_idx+m-1) % DBLOCK_SIZE == 0) {
								size_t dblock_row_ofs = (size_t)ub_idx * (size_t)max_query_len * (size_t)max_query_len / DBLOCK_SIZE + (size_t)max_query_len * (size_t)(ref_idx+m-1) / DBLOCK_SIZE + (size_t)query_idx;
								dblock_row[dblock_row_ofs] = HD;
							}
							//---------------------------------------------
							query_idx++;
							
						}

				

						//-------------- save dblock col -------------------
						if (((packed_query_idx+1)<<3)%DBLOCK_SIZE == 0 && packed_query_idx > 0) { // TODO: merge with the loop below? idk
							short2 tmp_HD;
							for (int ridx = 0; ridx < 8; ridx++) {
								tmp_HD.x = h[ridx+1];
								tmp_HD.y = f[ridx+1];
								size_t dblock_col_ofs = (size_t)ub_idx * (size_t)max_query_len * (size_t)max_query_len / DBLOCK_SIZE 
								+ (size_t)max_query_len * (size_t)((packed_query_idx+1)<<3) / DBLOCK_SIZE  
								+ (size_t)ref_idx + (size_t)ridx;
								dblock_col[dblock_col_ofs] = tmp_HD;
							}
						}
						//--------------------------------------------------


					}
					
					packed_query_idx++;
					
				}
				
				// write intermediate values to global buffers
				if (active) {	
					for (m = 1; m < 9; m++) {
						if ( ref_idx + m-1 < ref_len) {
							HD.x = h[m];
							HD.y = f[m];
							global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
						}
					}
					global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
				}
				
				finished_blocks+=warp_len;
			}

			__syncwarp();

			last_diag = (i+_cudaSliceWidth)<<3;
			prev_max_score = query_len+ref_len-1;
			
			__syncwarp();

			// If job is finished
			if (terminated) {
				total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
				if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
			}
			
			// Update the max score and its index to shared memory (used in Subwarp Rejoining)
			if (warp_id==1) shared_job[20+(warp_num&3)] = warp_num_block;
		
			__syncwarp();

			i += _cudaSliceWidth;

			/*Job wrap-up*/
			// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
			if (i >= total_anti_diags) {
				
				// Spill the results to GPU memory to be later moved to the CPU
				if (warp_id==0) {
					/* TODO: fix warp num block issue */
					if(warp_num_block < SHARED_SIZE && warp_num_block >= 0) {
						device_res->aln_score[ub_idx] = global_max_score[warp_num_block];//copy the max score to the output array in the GPU mem
						device_res->query_batch_end[ub_idx] = global_max_query_idx[warp_num_block];//copy the end position on query_batch sequence to the output array in the GPU mem
						device_res->target_batch_end[ub_idx] = global_max_ref_idx[warp_num_block];//copy the end position on target_batch sequence to the output array in the GPU mem
					}
			
					dropped[ub_idx] = terminated;
				}

				/*Subwarp Rejoining*/
				// The subwarp that has no job looks for new jobs by iterating over other subwarp's job
				for (m = 0; m < (32/const_warp_len); m++) {
					// if the selected job still has remainig diagonals
					if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
						// read the selected job's info
						total_anti_diags = shared_job[m];
						warp_num = ((warp_num>>2)<<2)+m;
						ub_idx = shared_job[16+m];

						packed_ref_batch_idx = shared_job[4+m];
						packed_query_batch_idx = shared_job[8+m];
						ref_len = shared_job[12+m];
						query_len = ref_len&65535;
						ref_len = ref_len>>16;
						packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
						packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);
						
						warp_num_block = shared_job[20+m];
					
						// reset the flag
						terminated = dropped[ub_idx];
						
						break;
					}
				}

			}

			__syncwarp();
			
			/*Subwarp Rejoining*/
			//Set the mask, warp length and thread id within the warp 
			same_threads = __match_any_sync(__activemask(), warp_num);
			warp_len = __popc(same_threads);
			warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;
			
			__syncwarp();

		}
		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];

		__syncwarp();

	}
	return;


}


__global__ void agatha_kernel_approx_dynamic_tb_offset(uint32_t *packed_query_batch, uint32_t *packed_ref_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, short2 *dblock_row, short2 *dblock_col, int n_tasks, uint32_t max_query_len, short2 *global_buffer_top,
	uint64_t* tb_ofs, bool* dropped, int bw)
{
/*Initial kernel setup*/

	// Initializing variables 
	int32_t i, k, m, l, y, e;
	int32_t ub_idx, job_idx, ref_idx, query_idx;
	short2 HD;
	int32_t temp_score;
	int slice_start, slice_end, finished_blocks, chunk_start, chunk_end;
	int packed_ref_idx, packed_query_idx;
	int total_anti_diags;
	register uint32_t packed_ref_literal, packed_query_literal; 
	bool active, terminated;
	int32_t packed_ref_batch_idx, packed_query_batch_idx, query_len, ref_len, packed_query_len, packed_ref_len;
	int diag_idx, temp, last_diag;

	uint64_t tb_offset;

	// Initializing max score and its idx
	__shared__ int32_t global_max_score[SHARED_SIZE]; 
	__shared__ int32_t global_max_ref_idx[SHARED_SIZE]; 
	__shared__ int32_t global_max_query_idx[SHARED_SIZE]; 
	__shared__ int32_t max_score[SHARED_SIZE]; 
	__shared__ int32_t max_ref_idx[SHARED_SIZE]; 
	__shared__ int32_t max_query_idx[SHARED_SIZE];
	bool max_block = true;

	int32_t prev_max_score = 0;
	
	// Setting constant values
	const short2 initHD = make_short2(MINUS_INF2, MINUS_INF2); //used to initialize short2
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; //thread ID within the entire kernel
	const int packed_len = 8; //number of bps (literals) packed into a single int32
	const int const_warp_len = 8; //number of threads per subwarp (before subwarp rejoining occurs)
	const int real_warp_id = threadIdx.x % 32; //thread ID within a single (full 32-thread) warp
	const int warp_per_kernel = (gridDim.x * blockDim.x) / const_warp_len; // number of subwarps. assume number of threads % const_warp_len == 0
	const int job_per_warp = n_tasks % warp_per_kernel ? (n_tasks / warp_per_kernel + 1) : n_tasks / warp_per_kernel; //number of jobs (alignments/tasks) needed to be done by a single subwarp
	const int job_per_query = max_query_len % const_warp_len ? (max_query_len / const_warp_len + 1) : max_query_len / const_warp_len; //number of a literal's initial score to fill per thread
	const int job_start_idx = (tid / const_warp_len)*job_per_warp; // the boundary of jobs of a subwarp 
	const int job_end_idx = (job_start_idx + job_per_warp) < n_tasks ? (job_start_idx + job_per_warp) : n_tasks; // the boundary of jobs of a subwarp
	const int total_shm = packed_len*(_cudaSliceWidth+1); // amount of shared memory a single thread uses

	// Arrays for saving intermediate values
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];

	// Global memory setup
	short2* global_buffer_left = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x);
	int32_t* global_buffer_topleft= (int32_t*)(global_buffer_left+max_query_len*(blockDim.x/8)*gridDim.x);
	short2* global_ub_idx = (short2*)(global_buffer_top+max_query_len*(blockDim.x/8)*gridDim.x*3);

	// Shared memory setup
	extern __shared__ int32_t shared_maxHH[];
	int32_t* shared_job = shared_maxHH+(blockDim.x/32)*total_shm*32+(threadIdx.x/32)*28;

	/* Setup values that will change after Subwarp Rejoining */
	int warp_len = const_warp_len;
	int warp_id = threadIdx.x % warp_len; // id of a thread in a subwarp 
	int warp_num = tid / warp_len;
	int warp_num_block = threadIdx.x / warp_len;
	// mask that is true for threads in the same subwarp
	unsigned same_threads = __match_any_sync(0xffffffff, warp_num);
	if (warp_id==0) shared_job[(warp_num&3)] = -1;

	bool test = warp_num < 4 && warp_id==0; // test for first warp..

	/* Iterating over jobs/alignments */
	for (job_idx = job_start_idx; job_idx < job_end_idx; job_idx++) {

		/*Uneven Bucketing*/
		// the first subwarp fetches a long sequence's idx, while the remaining subwarps fetch short sequences' idx
		ub_idx = ((job_idx&3)==0)? static_cast<uint16_t>(global_ub_idx[n_tasks-(job_idx>>2)-1].y): static_cast<uint16_t>(global_ub_idx[job_idx-(job_idx>>2)-1].y);

		// get target and query sequence information
		packed_ref_batch_idx = target_batch_offsets[ub_idx] >> 3; //starting index of the target_batch sequence
		packed_query_batch_idx = query_batch_offsets[ub_idx] >> 3;//starting index of the query_batch sequence
		query_len = query_batch_lens[ub_idx]; // query sequence length
		ref_len = target_batch_lens[ub_idx]; // reference sequence length 
		packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
		packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);//number of 32-bit words holding target_batch sequence

		// check if alignment in the chain is done 
		terminated = dropped[ub_idx];
		
		tb_offset = tb_ofs[ub_idx]; // TODO: check if this if correct

		/*Buffer Initialization*/
		// fill global buffer with initial value
		// global_buffer_top: used to store intermediate scores H and E in the horizontal strip (scores from the top)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_top[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_left: used to store intermediate scores H and F in the vertical strip (scores from the left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if ((l) < max_query_len) {
				k = -(_cudaGapOE + (_cudaGapExtend*(l)));
				global_buffer_left[warp_num*max_query_len + l] =  l <= bw? make_short2(k, k-_cudaGapOE):initHD;	
			}
		}
		// global_buffer_topleft: used to store intermediate scores H in the diagonal strip (scores from the top-left)
		for (i = 0; i < job_per_query; i++) {
			l = i*warp_len + warp_id;
			if (l < max_query_len) {
				k = -(_cudaGapOE+(_cudaGapExtend*(l*packed_len-1)));
				global_buffer_topleft[warp_num*max_query_len + l] = l==0? 0: (l*packed_len-1) <= bw? k: MINUS_INF2; 	
			}
		}

		// Initialize variables
		global_max_score[warp_num_block] = 0;
		global_max_ref_idx[warp_num_block] = 0;
		global_max_query_idx[warp_num_block] = 0;
		max_score[warp_num_block] = 0; 
		prev_max_score = 0;
		max_ref_idx[warp_num_block] = 0; 
		max_query_idx[warp_num_block] = 0;
		
		__syncwarp();

		i = 0; //chunk
		total_anti_diags = packed_ref_len + packed_query_len-1; //chunk
		
		/*Subwarp Rejoining*/
		//set shared memory that is used to maintain values for subwarp rejoining
		if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags;
		else if (warp_id==1) shared_job[4+(warp_num&3)] = packed_ref_batch_idx;
		else if (warp_id==2) shared_job[8+(warp_num&3)] = packed_query_batch_idx;
		else if (warp_id==3) shared_job[12+(warp_num&3)] = (ref_len<<16)+query_len;
		else if (warp_id==4) shared_job[16+(warp_num&3)] = ub_idx;

		same_threads = __match_any_sync(__activemask(), warp_num);

		__syncwarp();

		/*Main Alignment Loop*/
		while (i < total_anti_diags) {
			//if(test) printf("%d\t[%d] %d / %d %d warp num %d \n", ub_idx, warp_id, i, total_anti_diags, terminated, warp_num);

			// set boundaries for current slice
			slice_start = max(0, (i-packed_query_len+1));
			slice_start = max(slice_start, (i*packed_len + packed_len-1+1 - bw)/2/packed_len);
			slice_end = min(packed_ref_len-1, i+_cudaSliceWidth-1);
			slice_end = min(slice_end, ((i+_cudaSliceWidth-1)*packed_len + packed_len-1 + bw)/2/packed_len);
			finished_blocks = slice_start;

			if (slice_start > slice_end) {
			terminated = true;
			}

			while (!terminated && finished_blocks <= slice_end) {
				// while the entire chunk diag is not finished
				packed_ref_idx = finished_blocks + warp_id;
				packed_query_idx = i - packed_ref_idx;
				active = (packed_ref_idx <= slice_end);	//whether the current thread has cells to fill or not

				if (active) {
					ref_idx = packed_ref_idx << 3;
					query_idx = packed_query_idx << 3;

					// load intermediate values from global buffers
					p[1] = global_buffer_topleft[warp_num*max_query_len + packed_ref_idx];

					for (m = 1; m < 9; m++) {
						if ( (ref_idx + m-1) < ref_len) {
							HD = global_buffer_left[warp_num*max_query_len + ref_idx + m-1];
							h[m] = HD.x;
							f[m] = HD.y;
						} else {
							// if index out of bound of the score table 
							h[m] = MINUS_INF2;
							f[m] = MINUS_INF2;
						}

					}

					for (m=2;m<9;m++) {
						p[m] = h[m-1];
					}

					// Set boundaries for the current chunk
					chunk_start = (max(0, (packed_ref_idx*packed_len - bw)))/packed_len;
					chunk_end = min( packed_query_len-1, ( (packed_ref_idx*packed_len + packed_len -1 + bw)) /packed_len );
					packed_ref_literal = packed_ref_batch[packed_ref_batch_idx + packed_ref_idx];
				}

				// Compute the current chunk
				for (y = 0; y < _cudaSliceWidth; y++) {
					if (active && chunk_start <= packed_query_idx && packed_query_idx <= chunk_end) {
						packed_query_literal = packed_query_batch[packed_query_batch_idx + packed_query_idx]; 
						query_idx = packed_query_idx << 3;

						// set max index value if this thread has max cell in its 8*8 block
						if(((max_ref_idx[warp_num_block] >= ref_idx-1) && (max_ref_idx[warp_num_block] <= ref_idx+7)) || ((max_query_idx[warp_num_block] >= query_idx-1) && (max_query_idx[warp_num_block] <= query_idx+7)))
						max_block = true;
						else max_block = false;

						for (k = 28; k >= 0 && query_idx < query_len; k -= 4) {
							uint32_t qbase = (packed_query_literal >> k) & 15;	//get a base from query_batch sequence

							// load intermediate values from global buffers
							HD = global_buffer_top[warp_num*max_query_len + query_idx];
							h[0] = HD.x;
							e = HD.y;

							if (packed_query_idx == chunk_start || packed_query_idx == chunk_end) {
								if(max_block){
									#pragma unroll 8
									for (l = 28, m = 1; m < 9; l -= 4, m++) {
										CORE_COMPUTE_BOUNDARY_APPROX_MAX();
									}
								}	
								else{
									#pragma unroll 8
									for (l = 28, m = 1; m < 9; l -= 4, m++) {
										CORE_COMPUTE_BOUNDARY_APPROX();
									}
								}
							} else if(max_block){
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
									CORE_COMPUTE_APPROX_MAX();
								}
							} else{
								#pragma unroll 8
								for (l = 28, m = 1; m < 9; l -= 4, m++) {
								CORE_COMPUTE_APPROX();
								}
							}

							// write intermediate values to global buffers
							HD.x = h[m-1];
							HD.y = e;
							global_buffer_top[warp_num*max_query_len + query_idx] = HD;

							//--------------- save dblock row -------------
							if ((ref_idx+m-1) % DBLOCK_SIZE_D == 0) {
							size_t dblock_row_ofs = (size_t)tb_offset + (size_t)query_len * (size_t)((ref_idx+m-1) / DBLOCK_SIZE_D) + (size_t)query_idx;

							dblock_row[dblock_row_ofs] = HD;
						}
						//---------------------------------------------
						query_idx++;
					}

					//-------------- save dblock col -------------------
					if (((packed_query_idx+1)<<3)%DBLOCK_SIZE_D == 0) { 
						short2 tmp_HD;
						for (int ridx = 0; ridx < 8; ridx++) {
							tmp_HD.x = h[ridx+1];
							tmp_HD.y = f[ridx+1];
							
							size_t dblock_col_ofs = (size_t)tb_offset  
							+ (size_t)ref_len * (size_t)((packed_query_idx+1)<<3) / DBLOCK_SIZE_D  
							+ (size_t)ref_idx + (size_t)ridx;

							dblock_col[dblock_col_ofs] = tmp_HD;
						}
					}
				//--------------------------------------------------
				}

				packed_query_idx++;

			}

			// write intermediate values to global buffers
			if (active) {	
				for (m = 1; m < 9; m++) {
					if ( ref_idx + m-1 < ref_len) {
						HD.x = h[m];
						HD.y = f[m];
						global_buffer_left[warp_num*max_query_len + ref_idx + m-1] = HD;
					}
				}
				global_buffer_topleft[warp_num*max_query_len + packed_ref_idx] = p[1];
			}

			finished_blocks+=warp_len;
		}

		__syncwarp();

		last_diag = (i+_cudaSliceWidth)<<3;
		prev_max_score = query_len+ref_len-1;

		__syncwarp();

		// If job is finished
		if (terminated) { // This was for zdrop so maybe was never needed and never called?
			// this is to mark that the job is finished
			total_anti_diags = i; // set the total amount of diagonals as the current diagonal (to indicate that the job has finished)	
			if (warp_id==0) shared_job[(warp_num&3)] = total_anti_diags; //update this to shared memory as well (this will be used in Subwarp Rejoining as an indicator that the subwarp's job is done)
		}

		// Update the max score and its index to shared memory (used in Subwarp Rejoining)
		if (warp_id==1) shared_job[20+(warp_num&3)] = warp_num_block;

		__syncwarp();

		i += _cudaSliceWidth;

		/*Job wrap-up*/
		// If the job is done (either due to (1) meeting the termination condition (2) all the diagonals have been computed)
		if (i >= total_anti_diags) {

			// Spill the results to GPU memory to be later moved to the CPU
			if (warp_id==0) {
				device_res->aln_score[ub_idx] = global_max_score[warp_num_block];//copy the max score to the output array in the GPU mem
				device_res->query_batch_end[ub_idx] = global_max_query_idx[warp_num_block];//copy the end position on query_batch sequence to the output array in the GPU mem
				device_res->target_batch_end[ub_idx] = global_max_ref_idx[warp_num_block];//copy the end position on target_batch sequence to the output array in the GPU mem
				dropped[ub_idx] = terminated;
			}

			/*Subwarp Rejoining*/
			//The subwarp that has no job looks for new jobs by iterating over other subwarp's job
			for (m = 0; m < (32/const_warp_len); m++) {
				// if the selected job still has remainig diagonals
				if (shared_job[m] > i) { // possible because all subwarps sync after each diagonal block is finished
				// read the selected job's info
					total_anti_diags = shared_job[m];
					warp_num = ((warp_num>>2)<<2)+m;
					ub_idx = shared_job[16+m];

					packed_ref_batch_idx = shared_job[4+m];
					packed_query_batch_idx = shared_job[8+m];
					ref_len = shared_job[12+m];
					query_len = ref_len&65535;
					ref_len = ref_len>>16;
					packed_query_len = (query_len >> 3) + (query_len & 7 ? 1 : 0);
					packed_ref_len = (ref_len >> 3) + (ref_len & 7 ? 1 : 0);

					warp_num_block = shared_job[20+m];
					tb_offset = tb_ofs[ub_idx];
					
					// reset the flag
					terminated = dropped[ub_idx];
			
					break;
				}
			}

		}

		__syncwarp();

		/*Subwarp Rejoining*/
		//Set the mask, warp length and thread id within the warp 
		same_threads = __match_any_sync(__activemask(), warp_num);
		warp_len = __popc(same_threads);
		warp_id = __popc((((0xffffffff) << (threadIdx.x % 32))&same_threads))-1;

		__syncwarp();

		} // end of main alignment loop
		__syncwarp();
		/*Subwarp Rejoining*/
		//Reset subwarp and job related values for the next iteration
		warp_len = const_warp_len;
		warp_num = tid / warp_len;
		warp_id = tid % const_warp_len;
		ub_idx = shared_job[16+(warp_num&3)];
		warp_num_block = threadIdx.x / warp_len;

		__syncwarp();


	}

	return;


}


/* Gap right, left extension kernel (bw=1) */

#define KSW_CIGAR_MATCH  0
#define KSW_CIGAR_INS    1
#define KSW_CIGAR_DEL    2
#define KSW_CIGAR_N_SKIP 3

__device__
static inline void push_cigar(int *n_cigar, uint32_t *cigar, uint32_t op, int len)
{
	if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1]&0xf)) {
		cigar[(*n_cigar)++] = len<<4 | op;
	} else cigar[(*n_cigar)-1] += len<<4;
	return;
}

__global__
void mm2_kswz_extension(char* qseqs, char* tseqs, uint32_t* qseq_len, uint32_t* tseq_len, uint32_t* qseq_ofs, uint32_t* tseq_ofs,
						uint32_t* packed_tb_matrix, gasal_res_t *device_res, int* n_cigar, uint32_t* cigar, int n_task, uint32_t max_query_len, bool left){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t trace_dir[5000][3]; 

	for(int job_idx = tid; job_idx < n_task; job_idx += blockDim.x * gridDim.x){

		uint32_t qlen = qseq_len[tid];
		uint32_t tlen = tseq_len[tid];
		if(qlen <= 0 || tlen <= 0 || qlen > max_query_len || tlen > max_query_len) {
			n_cigar[tid] = 0; // TODO: also reset other part too & add dropped flag
			continue;
		}

		uint32_t qseq_offset = qseq_ofs[tid];
		uint32_t tseq_offset = tseq_ofs[tid];
		char* qseq = &qseqs[qseq_offset];
		char* tseq = &tseqs[tseq_offset];

		/* forward score computation */
		int r;

		// register for score computation
		int32_t h = MINUS_INF2; // current score - do we need this? maybe not...
		int32_t f; // f (from left)
		int32_t p[3] = {0,0,-(_cudaGapOE + _cudaGapExtend)}; // p (from diag)
		int32_t e[4] = {MINUS_INF2,- (_cudaGapOE * 2 + _cudaGapExtend),- (_cudaGapOE * 2 + _cudaGapExtend * 2),MINUS_INF2}; // e (from top)

		int32_t max_score = MINUS_INF2;
		uint32_t max_qpos = 0;
		uint32_t max_tpos = 0;

		// initialize registers

		for(int i = 0; i < tlen; i++){ // loop over tseq
			uint32_t tbase = tseq[i];
			f = (i==0)? - (_cudaGapOE * 2 + _cudaGapExtend) : MINUS_INF2; // initialize F : it comes from an out-of-bound cell
			e[3] = MINUS_INF2; // intialize rightmost E : it comes from an out-of-bound cell 
			for (int offset = -1; offset <= 1; offset++) {
				int j = i + offset;
				if (j < 0 || j >= qlen) continue;
				uint32_t qbase = qseq[j];

				int idx = offset + 1; 
				int e_idx = (idx + 1); // rolling window 
				int temp_score;
				
				// compute and update score
				DEV_GET_SUB_SCORE_GLOBAL(temp_score, qbase, tbase);
				temp_score += p[idx]; 
				uint32_t m_or_x = temp_score >= p[idx] ? 0 : 1;
				h = max(temp_score, f); 
				h = max(h, e[e_idx]); 
				trace_dir[i][idx] = 0;
				if(left){
					trace_dir[i][idx] |= (h == temp_score) ? m_or_x : ((f == temp_score) ? 3 : 2);
					trace_dir[i][idx] |= (temp_score - _cudaGapOE) > (f - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << 3;
					trace_dir[i][idx] |= (temp_score - _cudaGapOE) > (e[e_idx] - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << 2;
				}
				else {
					trace_dir[i][idx] |= h == temp_score ? m_or_x : (h == f ? (uint32_t)3 : (uint32_t)2); // 2 LSBs
					trace_dir[i][idx] |= (temp_score - _cudaGapOE) >= (f - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << 3;
					trace_dir[i][idx] |= (temp_score - _cudaGapOE) >= (e[e_idx] - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << 2;
				}
				f = max(temp_score- _cudaGapOE, f - _cudaGapExtend); // compute f for next (left) cell
				e[idx] = max(temp_score- _cudaGapOE, e[e_idx] - _cudaGapExtend); 
				p[idx] = h; 

				// update max score & check zdrop
				max_score = max(max_score, h);
				if(max_score == h){
					max_qpos = j;
					max_tpos = i;
				}
				// mm2_check_zdrop();
			}
		}
		device_res->aln_score[tid] = max_score;
	
		/* traceback phase */
		int i, j;
		i = max_tpos;
		j = max_qpos;

		if(i < 0 || j < 0 || i > max_query_len || j > max_query_len) return;
		int* n = &n_cigar[tid]; *n = 0;

		// traceback
		int state = 0;

		while (i >= 0 && j >= 0) {
			int direction = trace_dir[i][j-i+1];
			int force_state = -1;

			if(i > j + 1) force_state = 2; // bandwidth is 1
			if(i < j - 1) force_state = 3; 
	
			direction = force_state < 0? direction : 0;
	
			// for 1-stage gap cost
			if(state<=1) state = direction & 3;  
			else if(!(direction >> (state) & 1)) state = 0; 
			if(state<=1) state = direction & 3;  
			if (force_state >= 0) state = force_state; 
	
			switch(state) {
				case 0: // matched
				case 1: // mismatched
					push_cigar(n, &cigar[qseq_ofs[tid]], KSW_CIGAR_MATCH, 1);
					i--;
					j--;
				break;
				case 2: // from upper cell
					push_cigar(n, &cigar[qseq_ofs[tid]], KSW_CIGAR_DEL, 1);
					i--;
				break;
				case 3: // from left cell
					push_cigar(n, &cigar[qseq_ofs[tid]], KSW_CIGAR_INS, 1);
					j--;
				break;
				}
		}
		if (i >= 0) push_cigar(n, &cigar[qseq_ofs[tid]], KSW_CIGAR_DEL, i + 1); // first deletion
		if (j >= 0) push_cigar(n, &cigar[qseq_ofs[tid]], KSW_CIGAR_INS, j + 1); // first insertion
	
		if(!left){
			uint32_t tmp;
			for (i = 0; i < (*n)>>1; ++i) { // reverse CIGAR
				tmp = cigar[qseq_ofs[tid] + i];  // Store the original value
				cigar[qseq_ofs[tid] + i] = cigar[qseq_ofs[tid] + *n - 1 - i];  // Assign swapped value
				cigar[qseq_ofs[tid] + *n - 1 - i] = tmp;  // Complete the swap
			}
		}
	}
}


__global__
void mm2_kswz_extension_simd(char* qseqs, char* tseqs, uint32_t* qseq_len, uint32_t* tseq_len, uint32_t* qseq_ofs, uint32_t* tseq_ofs,
						uint32_t* packed_tb_matrix, gasal_res_t *device_res, int* n_cigar, uint32_t* cigar, int n_task, uint32_t max_query_len, bool left){
	// kernel to reproduce the minimap2 SIMD-operation
}