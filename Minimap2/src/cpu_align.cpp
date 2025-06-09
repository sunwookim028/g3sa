#include "cpu_align.h"

std::mutex log_mutex;
std::ofstream fallback_log("cpu_fallback.log", std::ios::out);

void fallback_worker(void* data, long i, int tid) {
    // fallback_worker_t* w = (fallback_worker_t*)data;
    // int read_id = w->read_ids[i];
    // mm_tbuf_t* tbuf = w->tbufs[tid];

    // const char* read_seq = w->buf->h_seqs + w->buf->h_ofs[read_id];
    // int read_len = w->buf->h_lens[read_id];

    // int n_regs = 0;
    // mm_reg1_t* regs = mm_map(w->mi, read_len, read_seq, &n_regs, tbuf, w->opt, NULL);
    fallback_worker_t* w = (fallback_worker_t*)data;
    const std::string& seq = w->reads[i]; // copied read sequence
    int read_id = w->read_ids[i];
    int read_len = seq.size();
    mm_tbuf_t* tbuf = w->tbufs[tid];

    int n_regs = 0;
    mm_reg1_t* regs = mm_map(w->mi, read_len, seq.c_str(), &n_regs, tbuf, w->opt, NULL);

    if (n_regs > 0 && regs[0].p && regs[0].p->n_cigar > 0) {
        std::string line;
        line.reserve(read_len * 5);

        line += "[read id:";
        line += std::to_string(read_id); // read id : temporary TODO: add read information
        line += "]\t";

        for (int j = 0; j < regs[0].p->n_cigar; ++j) {
            uint32_t c = regs[0].p->cigar[j];
            int len = static_cast<int>(c >> 4);
            char op = MM_CIGAR_STR[c & 0xf];
            line += std::to_string(len);
            line += op;
        }

        if (w->fout && w->fout_mutex) {
            std::lock_guard<std::mutex> lock(*w->fout_mutex);
            *(w->fout) << line << '\n';
        }
    }

    // log for debugging purposes
    {
        std::lock_guard<std::mutex> lock(log_mutex);

        fallback_log << "[Thread " << tid << "] Read " << read_id
                     << ", length=" << read_len
                     << ", seq=" << std::string(seq.c_str(), std::min(read_len, 30)) << "...\n";

        fallback_log << "[Thread " << tid << "] → " << n_regs << " mappings:\n";

        for (int j = 0; j < n_regs; ++j) {
            const mm_reg1_t& reg = regs[j];

            fallback_log << "  [r=" << reg.rid
             << ", s=" << reg.rs << ", e=" << reg.re
             << ", MAPQ=" << int(reg.mapq)
             //<< ", is_sec=" << reg.is_sec
             << ", p=" << (reg.p ? "yes" : "null")
             << ", n_cigar=" << (reg.p ? reg.p->n_cigar : 0) << "]\n";

            if (reg.p && reg.p->n_cigar > 0) {
                std::ostringstream cigar_str;
                for (int k = 0; k < reg.p->n_cigar; ++k) {
                    uint32_t c = reg.p->cigar[k];
                    cigar_str << (c >> 4) << "MIDNSHP=XB"[c & 0xf];
                }
                fallback_log << " CIGAR=" << cigar_str.str();
            }

            fallback_log << "\n";
        }
    }

    free(regs);
}

void run_cpu_fallback_batch(
    const std::vector<int>& read_ids,
    host_input_buffer* buf,
    mm_idx_t* index,
    mm_mapopt_t* opt,
    int n_threads
) {
    fallback_worker_t worker;
    worker.read_ids = const_cast<int*>(read_ids.data());
    worker.num_reads = read_ids.size();
    //worker.buf = buf;
    worker.mi = index;
    worker.opt = opt;
    worker.tbufs = new mm_tbuf_t*[n_threads];

    for (int i = 0; i < n_threads; ++i)
        worker.tbufs[i] = mm_tbuf_init();

    kt_for(n_threads, fallback_worker, &worker, worker.num_reads);

    for (int i = 0; i < n_threads; ++i)
        mm_tbuf_destroy(worker.tbufs[i]);
    delete[] worker.tbufs;
}
