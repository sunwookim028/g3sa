#ifndef CPU_ALIGN_H
#define CPU_ALIGN_H

#include <iostream>
#include <vector>
#include <mutex>
#include <sstream>
#include <iomanip>  // optional: for hex formatting
#include "kthread.h"
#include "mmpriv.h"
#include "host_mem.h"
#include "minimap.h"

// extern "C" {
//     #include "minimap.h"
// }

struct fallback_batch {
    std::vector<int> read_ids;
    std::vector<std::string> seqs;
    int buf_id;
};

// struct fallback_worker_t {
//     int* read_ids;
//     int num_reads;
//     host_input_buffer* buf;
//     mm_idx_t* mi;
//     mm_mapopt_t* opt;
//     mm_tbuf_t** tbufs;

//     std::ofstream* fout;
//     std::mutex* fout_mutex;
// };

struct fallback_worker_t {
    const mm_idx_t* mi;
    const mm_mapopt_t* opt;
    std::vector<std::string> reads; // copied sequences
    int* read_ids;
    int num_reads;
    std::ofstream* fout;
    std::mutex* fout_mutex;
    mm_tbuf_t** tbufs;
};

void fallback_worker(void* data, long i, int tid);
void run_cpu_fallback_batch(
    const std::vector<int>& read_ids,
    host_input_buffer* buf,
    mm_idx_t* index,
    mm_mapopt_t* opt,
    int n_threads
);
#endif
