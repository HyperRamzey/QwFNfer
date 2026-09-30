#pragma once
// io_uring read engine.
//
// This exists because the measured bottleneck on the target machine is not
// bandwidth but *granularity*. Demand-paging the expert weights through mmap
// yields 4.4 KiB average reads at queue depth ~4 and tops out near 0.30 GB/s.
// The same NVMe sustains 7.0 GB/s when asked for 640 KiB at queue depth 4 --
// which is exactly the size of one expert slice. So: explicit, batched,
// slice-sized, O_DIRECT reads.
//
// O_DIRECT also keeps the kernel page cache out of the way. We manage the RAM
// tier ourselves; letting the page cache mirror it would halve our effective
// capacity on a 30 GB machine.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "qwfn_platform.h"

#ifdef _WIN32
// On Windows the thread-pool backend (also the engine's Linux default) is
// the only backend; the ring pointer below is unused and stays null.
struct io_uring;   // never defined on this platform; the member exists only to
                   // keep the class layout identical across platforms
#else
struct io_uring;
#endif

namespace qwfn {

// The alignment the RAM tier's slots and the prefill staging are laid out to.
// A direct read needs offset, length and destination aligned to what the
// FILESYSTEM requires, not the device: NVMe reports 512, but btrfs (sectorsize
// 4096) serves anything not 4096-aligned through the page cache, silently, and
// measured on 2026-09-09 every expert read of this engine was buffered. The
// layout is therefore 4096 whenever every expert slice stride in the file is a
// page multiple, so the payload sits at a fixed offset per part and the read
// lands in place with no copy (the Q4 file); it is 512 otherwise (the Q3 file:
// IQ3_XXS slices step by 512 mod 4096), and the thread backend then reads a
// page-aligned window into a per-worker bounce buffer and copies the payload
// into the slot -- still a direct read, no page cache. Set by the engine before
// any layout is computed.
uint64_t dio_align();
void     set_dio_align(uint64_t a);
static constexpr uint64_t QWFN_DIO_PAGE = 4096;

inline uint64_t dio_align_down(uint64_t x) { return x & ~(dio_align() - 1); }
inline uint64_t dio_align_up  (uint64_t x) { return (x + dio_align() - 1) & ~(dio_align() - 1); }

// Bytes of slack between the start of an aligned read and the requested data.
inline uint32_t dio_pad(uint64_t offset) { return (uint32_t) (offset - dio_align_down(offset)); }

// Size of the aligned read needed to cover [offset, offset+nbytes).
inline uint32_t dio_padded_size(uint64_t offset, uint32_t nbytes) {
    return (uint32_t) (dio_align_up(dio_pad(offset) + nbytes));
}

struct io_request {
    int      shard   = 0;        // index into the paths passed to init()
    uint64_t offset  = 0;        // logical byte offset of the wanted data
    uint32_t nbytes  = 0;        // logical size of the wanted data
    void *   dst     = nullptr;  // dio_align()-aligned, >= dio_padded_size() bytes
    uint64_t tag     = 0;        // returned verbatim on completion
};

class io_engine {
public:
    io_engine() = default;
    ~io_engine();
    io_engine(const io_engine &) = delete;
    io_engine & operator=(const io_engine &) = delete;

    // Two backends behind one interface.
    //
    //   uring   : io_uring. On this filesystem io_uring_submit() turns out to
    //             execute the reads inline rather than queueing them, so a
    //             burst gets far less than the concurrency it asked for --
    //             measured 4.07 GB/s on the engine's 8-read burst.
    //   threads : a pool of workers doing blocking positional preadv. Real
    //             kernel-level parallelism; measured 5.30 GB/s on the same
    //             burst shape.
    //
    // Windows runs the threads backend only, over unbuffered overlapped reads
    // (FILE_FLAG_NO_BUFFERING): sector-aligned slice reads that bypass the
    // cache manager, the documented Windows equivalent of O_DIRECT.
    //
    // Windows' own IoRing (io_uring's SQ/CQ design, Win11 21H2+) was considered.
    //
    // This comment used to say "measured: thread pool 3.4-4.8 GB/s vs IoRing
    // 1.3-1.9 GB/s -- the port is 2.5-3.6x faster". THAT NUMBER WAS WRONG and is
    // retracted. No IoRing implementation was ever committed, so it had no
    // reproducible source, and when one was written (tools/qwfn_ioring.h,
    // driven by tools/qwfn_iobpio.cpp) it did not reproduce.
    //
    // MEASURED 2026-09-30, HP FX900 Pro 2TB, 16 GB file, random 640 KiB-1 MiB
    // slices, QD 1-64, modes interleaved, every mode byte-exact:
    //
    //   ioring/direct+wait   1.62 GB/s at QD 8 (+1.2%), 1.60 at QD 32 (-0.5%)
    //                        thread pool 1.60 / 1.61 on the same pattern
    //   service latency, QD 1, 2000 samples: ring p50 539 us, mean 541;
    //                                    pool p50 557 us, mean 559
    //
    // So a ring is not slower than this pool -- it is the same to within about
    // a percent, and marginally lower latency. There is no speed argument either
    // way. What there is instead is a shape argument, and it is the one that
    // decided it:
    //
    // The engine prefetches the next layer's experts *while the current layer
    // computes*, then reaps whatever has finished (reap(min_complete)). A
    // non-blocking SubmitIoRing does not give the kernel a reason to drain the
    // submission queue, and with nothing blocking, BuildIoRingReadFile eventually
    // fails with IORING_E_SUBMISSION_QUEUE_FULL -- measured, not reasoned:
    // every opportunistic-reap variant failed that way, on unbuffered and
    // buffered handles alike, while the blocking variant
    // (SubmitIoRing(n, INFINITE), i.e. waiting for N completions, which Windows
    // does support after all) was correct and fast. A ring port would have to
    // block, or poll, and blocking would give back the overlap the prefetch
    // exists to get.
    //
    // Preregistered buffers changed nothing measurable (11.96 vs 12.52 GB/s,
    // noise) and would be awkward here anyway: the RAM tier is one large arena
    // read through per-slot views, not standalone buffers.
    //
    // The submit/reap shape below is IoRing-like, so a future backend can slot
    // in without touching this interface.
    enum class backend { uring, threads };

    // queue_depth is the io_uring ring size / the worker count.
    //
    // bypass_io is Windows-only and opt-in; on Linux it is ignored. It asks for
    // FSCTL_MANAGE_BYPASS_IO on every handle, which is the platform's one
    // mechanism that takes work *out* of the read path (a noncached read skips
    // the filesystem, volume and storage filter stacks) rather than batching
    // the submission of it. It only exists on Windows 11 + NTFS + NVMe, only
    // for noncached reads, and is per file-open: any other handle on the same
    // file opened for cached or memory-mapped I/O suspends it until that handle
    // closes -- which this engine does, for the PLE table, on the same shards.
    //
    // MEASURED 2026-09-30 with tools/qwfn_iobpio.cpp, two NVMe volumes
    // (HP FX900 Pro 2TB, Intel SSDPEKNU512GZ), 16 GB and 12 GB files, 640 KiB
    // - 1 MiB random slices, QD 1-64, modes interleaved over 4 rounds, every
    // mode byte-exact: the kernel accepted it on every handle ("enabled on 1
    // handle(s)") and throughput moved by -0.2% to -1.1% (C:) and -0.9% to
    // -2.9% (D:) -- noise, never positive. Holding a read-only mapping of the
    // same file open, as the engine does for the PLE table, moved it by -0.3%
    // to +0.5%, which is what suspension predicts: it puts the read back on the
    // path it already had.
    //
    // The reason is visible in the same table and is the point: direct reads
    // plateau at 1.6 GB/s (C:) and 1.17 GB/s (D:) by QD 8 and do not move at
    // QD 16, 32 or 64. The device is the limit from QD 8, so there is no
    // per-operation overhead left to remove -- which is also why IoRing could
    // not have won here. Kept because it is cheap, and because the evidence for
    // not bothering is worth more than the flag.
    bool init(const std::vector<std::string> & paths, unsigned queue_depth,
              bool direct_io, std::string & err, backend be = backend::uring,
              bool bypass_io = false);

    backend which() const { return be_; }
    void shutdown();

    bool              bypass_requested() const { return bypass_req_; }
    bool              bypass_active()   const { return bypass_on_; }
    const std::string & bypass_note()    const { return bypass_note_; }

    // Queue reads. Returns the number accepted (short only if the ring is full).
    // Data for request i lands at dst + dio_pad(offset) when direct I/O is on,
    // and at dst when it is off.
    size_t submit(const io_request * reqs, size_t n);

    // Collect completions. Blocks until at least min_complete have arrived
    // (0 = purely opportunistic). Returns how many tags were written out.
    size_t reap(uint64_t * tags_out, size_t max_tags, size_t min_complete);

    size_t in_flight() const { return in_flight_; }
    bool   direct_io() const { return direct_; }

    // Offset within dst where the requested bytes actually begin.
    uint32_t payload_offset(uint64_t offset) const { return direct_ ? dio_pad(offset) : 0; }

    // Cumulative counters, for the benchmark and the runtime stats line.
    uint64_t stat_reads = 0, stat_bytes = 0, stat_errors = 0, stat_short = 0;
    double   stat_t_submit_syscall = 0;   // time inside io_uring_submit()
    double   stat_t_prep = 0;             // time building SQEs
    bool     registered_files = false;

private:
    backend          be_ = backend::threads;
    io_uring *       ring_ = nullptr;   // POSIX uring backend only; null on Windows

    // --- thread-pool backend ---
    struct job { int shard; uint64_t off; uint32_t len; void * dst; uint64_t tag; uint64_t ooff; uint32_t onb; };   // ooff/onb: the requested range, for the bounce path
    std::vector<std::thread>  workers_;
    std::deque<job>           q_;
    std::deque<uint64_t>      done_;
    std::mutex                mtx_;
    std::condition_variable   cv_work_, cv_done_;
    bool                      stop_ = false;
    void worker_loop();
    std::vector<int> fds_;
    bool             direct_ = true;
    bool             bounce_ = false;   // direct reads through a page-aligned per-worker buffer (512-byte slot layout)
    bool             bypass_req_ = false;   // bypass_io was asked for
    bool             bypass_on_  = false;   // the kernel accepted it on every handle
    std::string      bypass_note_;           // what the kernel said, or why it is unavailable
    size_t           in_flight_ = 0;
    unsigned         qd_ = 0;
    uint32_t         expect_[1024] = {};
    uint32_t         min_expect_ = 0;
};

// Page-aligned allocation helper for O_DIRECT destination buffers.
void * dio_alloc(size_t bytes);
void   dio_free(void * p);

// --- memory safety -------------------------------------------------------
// This engine's RAM tier is one large anonymous arena, and the target machine
// has 30 GB of RAM behind 30 GB of zram swap at vm.swappiness=150. Asking for
// more than the kernel can actually spare does not fail the allocation -- it
// gets compressed into zram, which itself consumes RAM, and the box OOMs.
// Every arena sizing therefore goes through clamp_to_available().

// MemAvailable from /proc/meminfo: the kernel's own estimate of what can be
// handed out without swapping. Returns 0 if it cannot be read.
uint64_t mem_available_bytes();

// Largest arena we are willing to take: `frac` of MemAvailable, minus a fixed
// headroom for activations, CUDA host buffers and the rest of the desktop.
// Never returns more than `want`.
size_t clamp_to_available(size_t want, double frac = 0.60,
                          size_t headroom = 3ull << 30);

} // namespace qwfn
