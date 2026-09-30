// qwfn-ioring -- a Windows IoRing reader, for qwfn_iobpio to A/B against the
// engine's thread pool.
//
// Why this exists: qwfn_io.h claims a measured 1.3-1.9 GB/s for "Windows IoRing
// v3, same pattern" against 3.4-4.8 GB/s for the thread pool, and no IoRing
// implementation was ever committed. A number nobody can reproduce is worse than
// no number, so this exists to either reproduce it or retire it.
//
// The design question it has to answer is not "is a ring faster" -- the engine's
// own comment already settles that -- but "why was it reported as 2.5-3.6x
// slower". The obvious suspect is that a ring read through a *buffered* handle
// is not the same operation as an unbuffered one: the cache manager then pays an
// allocation and a copy per 1 MiB, and per-operation cost becomes the bottleneck
// that the unbuffered path removed. So the reader is parameterised on the two
// things that could plausibly differ, and the benchmark crosses them:
//
//   direct          FILE_FLAG_NO_BUFFERING on the handle, or not
//   register_buffers  IORING_OP_READ into a preregistered buffer, or a raw
//                   pointer (which the kernel must probe per operation -- the
//                   exact work the ring was supposed to remove)
//   blocking_wait   SubmitIoRing(waitOperations = n) blocks until the burst is
//                   done, or submit is non-blocking and completions are reaped
//                   opportunistically, which is the engine's actual shape
//
// The engine's contract is reap(tags, max, min_complete): reap whatever has
// finished and keep computing. That maps onto a non-blocking submit plus a pop
// loop, so blocking_wait=false is the honest default here; blocking_wait=true
// is the best case a careful port could ask for, and the gap between the two is
// itself a finding.
#pragma once

#ifdef _WIN32

// This header pulls in <windows.h>, and whoever includes it may well be mid-file
// in C++ -- so the macros that break std::min / std::max have to be suppressed
// here rather than left to the includer to remember.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
// NTDDI_WIN10_NI: IoRing appeared in build 22000, and this is the gate
// ioringapi.h and ntioring_x.h test before exposing anything.
#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x0A000010
#endif

#include <windows.h>
#include <ioringapi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "qwfn_io.h"

namespace qwfn {

// A read request, in the same shape the engine's io_request uses.
struct io_slice { uint64_t off; uint32_t len; };

class ioring_reader {
public:
    ~ioring_reader() { close(); }
    ioring_reader() = default;
    ioring_reader(const ioring_reader &) = delete;
    ioring_reader & operator=(const ioring_reader &) = delete;

    // direct:           open the handle with FILE_FLAG_NO_BUFFERING
    // register_buffers: preregister the landing buffers once, address by index
    // blocking_wait:    SubmitIoRing(n, INFINITE) instead of a pop loop
    bool open(const std::string & path, unsigned qd, bool direct,
              bool register_buffers, bool blocking_wait, std::string & err) {
        close();
        direct_   = direct;
        reg_      = register_buffers;
        blocking_ = blocking_wait;

        // O_DIRECT's Windows equivalent. A ring read is still a ReadFile as far
        // as the storage stack is concerned; which of these two flags it carries
        // is the whole experiment.
        file_ = CreateFileW(std::filesystem::path(path).wstring().c_str(), GENERIC_READ,
                            FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            direct ? (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
                                   : FILE_FLAG_OVERLAPPED,
                            nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            char b[128] = {0};
            FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr, GetLastError(), 0, b, sizeof(b), nullptr);
            err = std::string("open failed: ") + b;
            return false;
        }

        // Ring sizes must be powers of two and the CQ is sized by the kernel.
        unsigned sq = 8;
        while (sq < qd * 2 && sq < 4096) sq <<= 1;
        IORING_CREATE_FLAGS f{};
        f.Required = IORING_CREATE_REQUIRED_FLAGS_NONE;
        // The builders validate their arguments on every call; the kernel still
        // reports per-SQE errors in the CQ, and we are about to make ~5000 calls
        // per pass, so the redundant user-mode checks are pure overhead.
        f.Advisory = IORING_CREATE_SKIP_BUILDER_PARAM_CHECKS;
        HRESULT hr = CreateIoRing(IORING_VERSION_1, f, sq, sq * 2, &ring_);
        if (FAILED(hr) || !ring_) {
            char b[64];
            snprintf(b, sizeof(b), "0x%08lX", (unsigned long) hr);
            err = std::string("CreateIoRing failed ") + b;
            close();
            return false;
        }
        if (!IsIoRingOpSupported(ring_, IORING_OP_READ)) {
            err = "IORING_OP_READ not supported on this build";
            close();
            return false;
        }
        GetIoRingInfo(ring_, &info_);

        // Register the file. IORING_OP_READ is issued against
        // IoRingHandleRefFromIndex(0), and an unregistered index is not a file:
        // the builder rejects it with ERROR_NOT_SUPPORTED. Registration is itself
        // an SQE, so it has to be submitted and reaped before the first read, and
        // its completion has to be *checked* -- a registration that fails
        // silently turns every later read into a CQE error.
        HANDLE h = file_;
        if (FAILED(BuildIoRingRegisterFileHandles(ring_, 1, &h, kTagRegisterFiles)) ||
            !submit_and_drain(1)) {
            err = "file registration failed: " + (note_.empty() ? std::string("?") : note_);
            close();
            return false;
        }
        return true;
    }

    // Preregister the landing buffers. Must be called once, after open(), with
    // the same addresses every later read_burst() will use -- the engine's RAM
    // tier is one large arena, so a real port would register per-slot views of it
    // rather than standalone buffers, and that is its own problem.
    bool register_landing_buffers(const std::vector<void *> & bufs, uint32_t bytes_each,
                                  std::string & err) {
        if (!ring_) { err = "not open"; return false; }
        std::vector<IORING_BUFFER_INFO> infos;
        infos.reserve(bufs.size());
        for (void * b : bufs) infos.push_back(IORING_BUFFER_INFO{ b, bytes_each });
        if (FAILED(BuildIoRingRegisterBuffers(ring_, (UINT32) infos.size(), infos.data(),
                                              kTagRegisterBuffers))) {
            err = "BuildIoRingRegisterBuffers failed";
            return false;
        }
        // The registration is an SQE with its own completion; an unchecked one
        // leaves every later read indexing a buffer that was never registered.
        if (!submit_and_drain(1)) {
            err = "buffer registration did not complete: " +
                  (note_.empty() ? std::string("?") : note_);
            return false;
        }
        return true;
    }

    // One burst: n reads, one landing buffer each, returning when all n have
    // completed. Same contract the engine's submit/reap pair provides.
    bool read_burst(const io_slice * s, unsigned n, void * const * bufs, std::string & err) {
        if (!ring_) { err = "not open"; return false; }
        for (unsigned i = 0; i < n; i++) {
            const IORING_BUFFER_REF bref =
                reg_ ? IoRingBufferRefFromIndexAndOffset(i, 0)
                     : IoRingBufferRefFromPointer(bufs[i]);
            const HRESULT hr = BuildIoRingReadFile(ring_, IoRingHandleRefFromIndex(0),
                                                   bref, s[i].len, s[i].off,
                                                   (UINT_PTR) (i + 1), IOSQE_FLAGS_NONE);
            if (FAILED(hr)) {
                char b[64];
                snprintf(b, sizeof(b), "BuildIoRingReadFile 0x%08lX", (unsigned long) hr);
                err = b;
                return false;
            }
        }
        if (!submit_and_drain(n)) {
            // A drain can fail without the builder failing: a CQE carrying an
            // error status is a failed read, not a slow one. Say so, or the
            // caller reports a failure with no reason attached.
            err = "submit/drain: " + (note_.empty() ? std::string("CQE error") : note_);
            return false;
        }
        return true;
    }

    void close() {
        if (ring_) { CloseIoRing(ring_); ring_ = nullptr; }   // not CloseHandle
        if (file_ != INVALID_HANDLE_VALUE) { CloseHandle(file_); file_ = INVALID_HANDLE_VALUE; }
    }

    uint64_t errors() const { return errors_; }
    uint64_t bytes()  const { return bytes_; }
    uint64_t reads()  const { return reads_; }
    const std::string & note() const { return note_; }
    bool direct() const { return direct_; }
    bool registered() const { return reg_; }
    bool blocking() const { return blocking_; }
    UINT32 sq_size() const { return info_.SubmissionQueueSize; }

    // Completion reaping, reported separately because it is the part of a ring
    // port that the thread pool gets for free from GetOverlappedResult.
    double   t_submit = 0, t_reap = 0;
    uint64_t empty_pops = 0;      // CQ was empty and we had to wait on it
    const std::string & first_error() const { return first_err_; }

private:
    static constexpr UINT_PTR kTagRegisterFiles    = 0xF11E0001ull;
    static constexpr UINT_PTR kTagRegisterBuffers = 0xB0FE0001ull;

    // Submit everything queued, then collect `want` completions.
    //
    // On the blocking path SubmitIoRing can return STATUS_TIMEOUT (0x102) when
    // the wait elapses, which as an HRESULT is a success code, so FAILED() alone
    // is the right test and the timeout needs no special case. It is not named
    // in any user-mode header, so it is not spelled out here either.
    bool submit_and_drain(unsigned want) {
        const auto t0 = std::chrono::steady_clock::now();
        UINT32 submitted = 0;
        const HRESULT hr = SubmitIoRing(ring_, blocking_ ? want : 0,
                                        blocking_ ? INFINITE : 0, &submitted);
        if (FAILED(hr)) {
            char b[64];
            snprintf(b, sizeof(b), "0x%08lX", (unsigned long) hr);
            note_ = std::string("SubmitIoRing failed ") + b;
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        t_submit += std::chrono::duration<double>(t1 - t0).count();

        // PopIoRingCompletion is non-blocking: it fails when the CQ is empty.
        // The engine's reap(min_complete) blocks instead, so an opportunistic
        // port has to wait here, and how much that costs is the interesting
        // number -- hence counting the empty pops rather than hiding them.
        unsigned got = 0;
        bool     clean = true;
        while (got < want) {
            IORING_CQE cqe{};
            if (FAILED(PopIoRingCompletion(ring_, &cqe))) {
                ++empty_pops_;
                YieldProcessor();
                continue;
            }
            ++got;
            if (!account(cqe)) clean = false;
        }
        const auto t2 = std::chrono::steady_clock::now();
        t_reap += std::chrono::duration<double>(t2 - t1).count();
        // A failed CQE is not a slow read, it is a wrong read: report it rather
        // than let the caller go and compare whatever the buffer happens to hold.
        return clean;
    }

    bool account(const IORING_CQE & cqe) {
        if (FAILED(cqe.ResultCode)) {
            ++errors_;
            if (first_err_.empty()) {
                char b[64];
                snprintf(b, sizeof(b), "CQE 0x%08lX", (unsigned long) cqe.ResultCode);
                first_err_ = b;
                if (note_.empty()) note_ = first_err_;
            }
            return false;
        }
        bytes_ += (uint64_t) cqe.Information;
        reads_ += 1;
        return true;
    }

    HIORING  ring_ = nullptr;
    HANDLE  file_ = INVALID_HANDLE_VALUE;
    IORING_INFO info_{};
    bool    direct_ = false, reg_ = false, blocking_ = false;
    uint64_t errors_ = 0, bytes_ = 0, reads_ = 0, empty_pops_ = 0;
    std::string note_, first_err_;
};

} // namespace qwfn

#endif // _WIN32
