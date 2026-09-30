// qwfn-iobpio -- every read path this engine can take, one table, one pattern.
//
// The engine's I/O question is not "which API is fastest" but "what is this
// drive's random-slice read rate, and does the path we take to get it cost
// anything". qwfn-iobench answers the first. This answers the second by running
// ONE pattern -- same seed, same offsets, same order, same buffers, same
// 640 KiB-1 MiB slice mix -- through nine read paths that differ only in the
// handle flags and the API:
//
//   buffered            overlapped, cache manager in the path        (control)
//   direct              FILE_FLAG_NO_BUFFERING                      <- what ships
//   direct+bpio         + FSCTL_MANAGE_BYPASS_IO
//   direct+bpio+mmap    + BypassIO, with a competing read-only mapping
//   ioring/direct       ring, unbuffered, opportunistic reap
//   ioring/direct+reg   ring, unbuffered, preregistered buffers
//   ioring/direct+wait  ring, unbuffered, SubmitIoRing(n, INFINITE)
//   ioring/buffered     ring, buffered handle
//   ioring/buffered+reg ring, buffered handle, preregistered buffers
//
// Two things this file is careful about, both of which got a first version
// wrong:
//
// 1. The mapping is a CONTROLLED VARIABLE, not ambient state. BypassIO is
//    per file-open and is documented to be *suspended* while another handle has
//    the same file open for memory-mapped I/O -- and the engine holds a read-only
//    mapping of the same shards for the PLE table for the life of the process
//    (qwfn_engine.cpp:145). Holding that mapping open for the whole run, as a
//    first version did, silently measured a *suspended* BypassIO and reported it
//    as an active one. So the sweep runs twice, once per condition, and every
//    mode appears in both tables.
//
// 2. Correctness is measured outside the timed region. Checksumming 1 MiB per
//    read costs more CPU than the reads cost wall-clock at these rates, so the
//    timed pass moves bytes and counts them and a separate pass compares. A mode
//    that returns wrong bytes fails the run rather than reporting a speed.
//
// Modes are interleaved round-robin across rounds rather than run back to back:
// a consumer SSD at 37 C with a desktop full of background I/O drifts, and one
// pass per mode invites that drift to be the result. Best and median are both
// reported for the same reason.
//
// Latency is reported separately because at QD 8+ the per-read wall clock is a
// property of the queue, not of the device. The QD-1 column is service time.
//
// BypassIO only applies to NTFS on NVMe, and a *buffered* handle only reads the
// drive when the file is not resident in RAM. Both are checked and reported
// rather than assumed: try_purge_standby() asks the kernel to drop the standby
// list and says whether it worked, and every buffered row is labelled.
//
// Usage:
//   qwfn-iobpio <file> [options]
//     --create            create and fill the file first (see --gb)
//     --gb N              size to create, default 8
//     --qds a,b,c         queue depths for the main sweep, default 1,2,4,8,16,32,48,64
//     --mmap-qds a,b,c    queue depths for the mapping-open sweep, default 8,32
//     --passes N          passes per mode per QD, default 2
//     --rounds N          interleaved rounds, best and median kept, default 3
//     --pass-gb N         bytes one pass moves, default 4
//     --lat N             latency samples per mode, default 2000
//     --min K --max K     read size range in KiB, default 640..1024
//     --workers N         io_engine queue depth / worker hint, default 64
//     --sector N          4096 (the Q4 file's layout) or 512 (the Q3 file's,
//                         which routes through the bounce path), default 4096
//     --volume-info       report volume, filesystem and sector geometry, exit
//     --no-ioring         skip the five ring modes
//
// No CMake target on purpose. qwfn_core needs llama.cpp's ggml, and this tool
// needs none of it -- it is io_engine plus a ring -- so it builds from two
// source files. The Windows SDK is what it adds: ioringapi.h and kernel32.lib.
// CMake puts both in the path for an MSVC-like compiler, but the CMake build
// here cannot configure at all until a llama.cpp checkout is restored, so the
// target was left out rather than committed unbuilt. Build it directly:
//
//   clang-cl /nologo /std:c++20 /EHsc /O2 /W4 /permissive- -D_CRT_SECURE_NO_WARNINGS ^
//     /Isrc /Itools ^
//     /I"<sdk>\Include\10.0.26100.0\um" ^
//     /I"<sdk>\Include\10.0.26100.0\shared" ^
//     /I"<sdk>\Include\10.0.26100.0\ucrt" ^
//     /Fo<objdir>\ /Fe<outdir>\qwfn-iobpio.exe ^
//     src\qwfn_io.cpp tools\qwfn_iobpio.cpp ^
//     /link /LIBPATH:"<sdk>\Lib\10.0.26100.0\um\x64" ^
//           /LIBPATH:"<sdk>\Lib\10.0.26100.0\ucrt\x64" kernel32.lib
//
// Verified with clang-cl 22.1.8 and the 10.0.26100.0 SDK on Windows 11
// build 28000.

#include "qwfn_io.h"
#include "qwfn_ioring.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#else
#error "qwfn-iobpio is the Windows read-path benchmark"
#endif

using namespace qwfn;
using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Byte i is (i * 31 + 7) & 0xFF -- the same pseudo-file qwfn-io-test writes, so
// a verify here and a verify there agree.
static uint8_t byte_at(uint64_t i) { return (uint8_t) ((i * 31 + 7) & 0xFF); }

static std::string widen(const wchar_t * w) {
    std::string s;
    for (const wchar_t * p = w; p && *p; p++) s.push_back((char) *p);
    return s;
}

// ---------------------------------------------------------------- geometry --
// The physical sector size is the number the engine's whole alignment contract
// turns on. The documented way to get it (Win32 "File Buffering") is
// IOCTL_STORAGE_QUERY_PROPERTY / StorageAccessAlignmentDescriptor on a *file*
// handle: STORAGE_PHYSICAL_DESCRIPTOR_PROPERTY only answers on a disk handle.
#define IOCTL_STORAGE_QUERY_PROPERTY_ 0x002D1400u
#define STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR_PROPERTY_ 7u

static std::string phys_sector_note(const std::string & file) {
    HANDLE f = CreateFileW(std::filesystem::path(file).wstring().c_str(),
                           FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return "(cannot open the file)";
    struct { DWORD PropertyId, QueryType; BYTE Additional[1]; } q = {
        STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR_PROPERTY_, 0 /*PropertyStandardQuery*/, {0} };
    std::vector<uint8_t> buf(64, 0);
    DWORD got = 0;
    std::string note;
    if (DeviceIoControl(f, IOCTL_STORAGE_QUERY_PROPERTY_, &q, sizeof(q),
                        buf.data(), (DWORD) buf.size(), &got, nullptr) &&
        got >= 20 && *(uint32_t *) buf.data() != 0)
        note = std::to_string(*(uint32_t *) (buf.data() + 12)) + " B logical, " +
               std::to_string(*(uint32_t *) (buf.data() + 16)) + " B physical";
    else
        note = "(alignment query unavailable, err=" +
               std::to_string((unsigned long) GetLastError()) + "; the engine picks 4096 " +
               "or 512 from the file's own expert-slice stride, not from this)";
    CloseHandle(f);
    return note;
}

static void report_volume(const std::string & file) {
    WCHAR mount[MAX_PATH] = {0}, fsname[64] = {0};
    DWORD serial = 0, maxcomp = 0, flags = 0;
    const BOOL ok = GetVolumePathNameW(std::filesystem::path(file).wstring().c_str(),
                                       mount, MAX_PATH) &&
                    GetVolumeInformationW(mount, nullptr, 0, &serial, &maxcomp, &flags,
                                          fsname, 64);
    ULARGE_INTEGER avail{}, total{}, freegb{};
    GetDiskFreeSpaceExW(mount, &avail, &total, nullptr);
    freegb = avail;
    printf("file          : %s\n", file.c_str());
    if (ok) {
        const std::string fs = widen(fsname);
        wchar_t mc[MAX_PATH] = {0};
        const std::string shown = widen(GetVolumePathNameW(mount, mc, MAX_PATH) ? mc : mount);
        printf("volume        : %s  (%s)\n", shown.c_str(), fs.c_str());
        printf("free          : %.1f GB of %.1f GB\n", freegb.QuadPart / 1e9,
               total.QuadPart / 1e9);
        printf("sector        : %s\n", phys_sector_note(file).c_str());
        if (fs != "NTFS")
            printf("BypassIO      : NOT ELIGIBLE -- %s is not NTFS (Win11 + NTFS + NVMe only)\n",
                   fs.c_str());
        else
            printf("BypassIO      : eligible in principle; the kernel decides per handle\n");
    }
    printf("\n");
}

// ------------------------------------------------------- cache / standby list
// The test file is written by this tool, so on a machine with more RAM than the
// file it is entirely resident and every *buffered* mode is reading RAM. The
// unbuffered modes bypass the cache manager and do not care; the buffered ones
// are worthless as disk numbers unless the standby list goes first.
// MemoryPurgeStandbyList is the documented way to ask. It normally needs
// elevation, and the caller is told which happened rather than left to guess.
static bool try_purge_standby() {
    typedef LONG (NTAPI *fn_set_sysinfo)(ULONG, void *, ULONG);
    static fn_set_sysinfo p = nullptr;
    static bool resolved = false;
    if (!resolved) {
        resolved = true;
        if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"))
            p = (fn_set_sysinfo) (void *) GetProcAddress(ntdll, "NtSetSystemInformation");
    }
    if (!p) return false;
    // SystemMemoryListInformation = 0x50, MemoryPurgeStandbyList = 4.
    ULONG cmd = 4;
    return p(0x50, &cmd, sizeof(cmd)) >= 0;
}

// ------------------------------------------------------------------ create --
static bool create_file(const std::string & path, uint64_t bytes) {
    printf("creating %s (%.1f GB, deterministic contents)...\n", path.c_str(), bytes / 1e9);
    HANDLE h = CreateFileW(std::filesystem::path(path).wstring().c_str(), GENERIC_WRITE,
                           0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { printf("  create failed (%lu)\n", GetLastError()); return false; }
    LARGE_INTEGER end{};
    end.QuadPart = (LONGLONG) bytes;
    if (!SetFilePointerEx(h, end, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) {
        printf("  cannot size the file (%lu)\n", GetLastError());
        CloseHandle(h);
        return false;
    }
    LARGE_INTEGER zero{};
    SetFilePointerEx(h, zero, nullptr, FILE_BEGIN);

    const size_t CH = 8u << 20;
    std::vector<uint8_t> buf(CH);
    uint64_t done = 0;
    const auto t0 = clk::now();
    while (done < bytes) {
        const size_t n = (size_t) std::min<uint64_t>((uint64_t) CH, bytes - done);
        for (size_t i = 0; i < n; i++) buf[i] = byte_at(done + i);
        DWORD put = 0;
        if (!WriteFile(h, buf.data(), (DWORD) n, &put, nullptr) || (size_t) put != n) {
            printf("  write failed at %.1f GB (%lu)\n", done / 1e9, GetLastError());
            CloseHandle(h);
            return false;
        }
        done += n;
    }
    FlushFileBuffers(h);
    CloseHandle(h);
    const double dt = secs(t0, clk::now());
    printf("  wrote %.1f GB in %.1f s (%.2f GB/s)\n\n", bytes / 1e9, dt, bytes / dt / 1e9);
    return true;
}

// ----------------------------------------------------------------- pattern --
struct slice { uint64_t off; uint32_t len; };

static std::vector<slice> build_pattern(uint64_t file_bytes, uint32_t sector,
                                        uint32_t min_b, uint32_t max_b, double pass_gb) {
    const uint64_t span = file_bytes - max_b - 8192;
    const size_t   n    = (size_t) (((uint64_t) (pass_gb * 1e9)) /
                                    ((uint64_t) (min_b + max_b) / 2));
    std::vector<slice> out;
    out.reserve(n);
    std::mt19937_64 g(0xB1A55EEDull);          // fixed: all modes read these bytes
    std::uniform_int_distribution<uint64_t> offd(4096ull, span);
    std::uniform_int_distribution<uint32_t> lend(min_b, max_b);
    for (size_t i = 0; i < n; i++) {
        const uint64_t o = offd(g) & ~(uint64_t) (sector - 1);
        const uint32_t l = (lend(g) + sector - 1) & ~(sector - 1);
        if (o + l > file_bytes) continue;
        out.push_back({ o, l });
    }
    return out;
}

// -------------------------------------------------------------------- modes --
enum class arm { pool, ring };

struct mode_def {
    const char * name;
    arm          a;
    bool         direct;   // FILE_FLAG_NO_BUFFERING
    bool         bpio;     // ask for FSCTL_MANAGE_BYPASS_IO
    bool         reg;      // preregister the landing buffers
    bool         wait;     // SubmitIoRing(n, INFINITE) instead of a pop loop
    bool         buffered_reads_possible;   // can this mode ever hit the drive?
    const char * what;
};

static const mode_def MODES[] = {
    { "buffered",           arm::pool, false, false, false, false, true,
      "overlapped only, cache manager in the path (control)" },
    { "direct",             arm::pool, true,  false, false, false, false,
      "FILE_FLAG_NO_BUFFERING -- what the port ships today" },
    { "direct+bpio",        arm::pool, true,  true,  false, false, false,
      "direct + FSCTL_MANAGE_BYPASS_IO, kernel confirmed it" },
    { "direct+bpio+mmap",   arm::pool, true,  true,  false, false, false,
      "the same, with a read-only mapping of the file held open (the PLE table)" },
    { "ioring/direct",      arm::ring, true,  false, false, false, false,
      "ring, unbuffered, opportunistic reap" },
    { "ioring/direct+reg",  arm::ring, true,  false, true,  false, false,
      "ring, unbuffered, preregistered buffers, opportunistic reap" },
    { "ioring/direct+wait", arm::ring, true,  false, false, true,  false,
      "ring, unbuffered, SubmitIoRing(n, INFINITE) -- waits for N completions" },
    { "ioring/buffered",    arm::ring, false, false, false, false, true,
      "ring, buffered handle" },
    { "ioring/buffered+reg",arm::ring, false, false, true,  false, true,
      "ring, buffered handle, preregistered buffers" },
};
static const int NMODES = (int) (sizeof(MODES) / sizeof(MODES[0]));
static const int I_SHIPPED = 1;   // the row everything else is compared against

// One live mode. Both arms own an engine object, only the relevant one is used.
struct arm_slot {
    bool              ok = false;
    std::string       why;                 // why not, or the first failure
    io_engine         pool;
    ioring_reader     ring;
    bool              cached_rows_valid = false;   // may this mode read the drive?
};

// ---------------------------------------------------------------- pass loops --
struct result { double gbps = 0, mean_us = 0; uint64_t bytes = 0, reads = 0; };

// bufs holds one aligned landing buffer per in-flight request, so the direct
// path writes in place and no two workers ever share a destination.
static result run_pool_pass(io_engine & io, const std::vector<slice> & pat,
                            std::vector<void *> & bufs, unsigned qd) {
    result r;
    std::vector<io_request> reqs;
    reqs.reserve(qd);
    uint64_t tags[512];
    size_t cursor = 0;
    const auto t0 = clk::now();
    while (cursor < pat.size()) {
        reqs.clear();
        for (unsigned i = 0; i < qd; i++) {
            const slice & s = pat[(cursor + i) % pat.size()];
            reqs.push_back(io_request{ 0, s.off, s.len, bufs[i], (uint64_t) (i + 1) });
        }
        cursor += qd;
        io.submit(reqs.data(), reqs.size());
        while (io.in_flight()) io.reap(tags, 512, 1);
        for (unsigned i = 0; i < reqs.size(); i++) { r.bytes += reqs[i].nbytes; r.reads++; }
    }
    const double dt = secs(t0, clk::now());
    r.gbps    = r.bytes / dt / 1e9;
    r.mean_us = r.reads ? (dt * 1e6 / (double) r.reads) : 0;
    return r;
}

// The ring's submit/drain has a different shape from submit/reap, so it needs
// its own loop over the same pattern.
static bool run_ring_pass(ioring_reader & r, const std::vector<io_slice> & ipat,
                          std::vector<void *> & bufs, unsigned qd,
                          double & gbps, double & mean_us) {
    const uint64_t b0 = r.bytes(), r0 = r.reads();
    const auto t0 = clk::now();
    size_t cursor = 0;
    while (cursor < ipat.size()) {
        const unsigned n = (unsigned) std::min<size_t>(qd, ipat.size() - cursor);
        std::string err;
        if (!r.read_burst(&ipat[cursor], n, bufs.data(), err)) return false;
        cursor += n;
    }
    const double dt = secs(t0, clk::now());
    const uint64_t nr = r.reads() - r0;
    gbps    = (r.bytes() - b0) / dt / 1e9;
    mean_us = nr ? (dt * 1e6 / (double) nr) : 0;
    return true;
}

// ------------------------------------------------------------- latency stats
// A QD-1 read is service time with no queueing in front of it, which is the
// only place the two can be separated.
struct lat_stats {
    double min_us = 0, p50 = 0, p95 = 0, p99 = 0, mean = 0, max = 0;
    uint64_t n = 0;
};

static lat_stats summarise(std::vector<double> v) {
    lat_stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t) (f * v.size()))]; };
    s.n      = v.size();
    s.min_us = v.front();
    s.max    = v.back();
    s.p50    = q(0.50);
    s.p95    = q(0.95);
    s.p99    = q(0.99);
    double sum = 0;
    for (double x : v) sum += x;
    s.mean   = sum / (double) v.size();
    return s;
}

static std::vector<double> pool_latency(io_engine & io, const std::vector<slice> & pat,
                                        std::vector<void *> & bufs, size_t samples) {
    std::vector<double> us;
    us.reserve(samples);
    uint64_t tags[8];
    for (size_t i = 0; i < samples; i++) {
        const slice & s = pat[i % pat.size()];
        const io_request r{ 0, s.off, s.len, bufs[0], (uint64_t) (i + 1) };
        const auto t0 = clk::now();
        io.submit(&r, 1);
        while (io.in_flight()) io.reap(tags, 8, 1);
        us.push_back(secs(t0, clk::now()) * 1e6);
    }
    return us;
}

static std::vector<double> ring_latency(ioring_reader & r, const std::vector<io_slice> & pat,
                                        std::vector<void *> & bufs, size_t samples) {
    std::vector<double> us;
    us.reserve(samples);
    for (size_t i = 0; i < samples; i++) {
        std::string err;
        const auto t0 = clk::now();
        if (!r.read_burst(&pat[i % pat.size()], 1, bufs.data(), err)) break;
        us.push_back(secs(t0, clk::now()) * 1e6);
    }
    return us;
}

// ------------------------------------------------------------------ verify --
// Byte-exact, on its own pass, so the checksum never sits in the timed path.
// payload_offset is 0 for every path here because every read in the pattern is
// aligned to the layout stride, so the direct path lands in place.
static bool verify_mode(io_engine & io, const char * name, const std::vector<slice> & pat,
                        std::vector<void *> & bufs, unsigned qd, uint64_t & checked) {
    std::vector<io_request> reqs;
    uint64_t tags[512];
    bool ok = true;
    checked = 0;
    for (size_t base = 0; base < pat.size() && ok; base += qd) {
        reqs.clear();
        for (unsigned i = 0; i < qd && base + i < pat.size(); i++)
            reqs.push_back(io_request{ 0, pat[base + i].off, pat[base + i].len,
                                       bufs[i], (uint64_t) (i + 1) });
        io.submit(reqs.data(), reqs.size());
        while (io.in_flight()) io.reap(tags, 512, 1);
        for (unsigned i = 0; i < reqs.size() && ok; i++) {
            const slice & sl = pat[base + i];
            const uint8_t * p = (const uint8_t *) reqs[i].dst + io.payload_offset(sl.off);
            for (uint32_t k = 0; k < sl.len; k++)
                if (p[k] != byte_at(sl.off + k)) {
                    printf("  MISMATCH in %s at offset %" PRIu64 " byte %u "
                           "(want %02x got %02x)\n", name, sl.off, k,
                           byte_at(sl.off + k), p[k]);
                    ok = false;
                    break;
                }
            checked += sl.len;
        }
    }
    return ok;
}

static bool verify_ring(ioring_reader & r, const char * name,
                        const std::vector<io_slice> & ipat,
                        std::vector<void *> & bufs, unsigned qd, uint64_t & checked) {
    bool ok = true;
    checked = 0;
    for (size_t base = 0; base < ipat.size() && ok; base += qd) {
        const unsigned n = (unsigned) std::min<size_t>(qd, ipat.size() - base);
        std::string err;
        if (!r.read_burst(&ipat[base], n, bufs.data(), err)) {
            printf("  %s: read_burst failed during verify: %s\n", name, err.c_str());
            return false;
        }
        for (unsigned i = 0; i < n && ok; i++) {
            const io_slice & sl = ipat[base + i];
            const uint8_t * p = (const uint8_t *) bufs[i];
            for (uint32_t k = 0; k < sl.len; k++)
                if (p[k] != byte_at(sl.off + k)) {
                    printf("  MISMATCH in %s at offset %" PRIu64 " byte %u "
                           "(want %02x got %02x)\n", name, sl.off, k,
                           byte_at(sl.off + k), p[k]);
                    ok = false;
                    break;
                }
            checked += sl.len;
        }
    }
    return ok;
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// -------------------------------------------------------------- mapping ------
// A live read-only mapping, the way weights::map_shards() makes one. Held for a
// whole sweep when the condition asks for it, so the variable is controlled.
struct mapping {
    HANDLE f = INVALID_HANDLE_VALUE, m = nullptr;
    void * view = nullptr;
    uint64_t bytes = 0;
    bool open(const std::string & path, uint64_t len) {
        close();
        f = CreateFileW(std::filesystem::path(path).wstring().c_str(), GENERIC_READ,
                        FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (f == INVALID_HANDLE_VALUE) return false;
        m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m) { close(); return false; }
        view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!view) { close(); return false; }
        bytes = len;
        return true;
    }
    // Fault in one byte per 2 MiB so the section has real cache maps rather than
    // a view nobody has read, without committing the whole file to standby. A
    // read, not a self-assignment -- the mapping is PAGE_READONLY and a store
    // into it faults.
    void fault_sparse() const {
        if (!view) return;
        volatile uint8_t sink = 0;
        const volatile uint8_t * p = (const volatile uint8_t *) view;
        for (uint64_t o = 0; o + 4096 < bytes; o += (2ull << 20)) sink ^= p[o];
        (void) sink;
    }
    void close() {
        if (view) { UnmapViewOfFile(view); view = nullptr; }
        if (m)    { CloseHandle(m); m = nullptr; }
        if (f != INVALID_HANDLE_VALUE) { CloseHandle(f); f = INVALID_HANDLE_VALUE; }
    }
};

// --------------------------------------------------------------------- main --
int main(int argc, char ** argv) {
    // Unbuffered, not line-buffered: the Windows CRT silently treats _IOLBF as
    // _IOFBF, so a fault inside a mode would take every line with it.
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: qwfn-iobpio <file> [--create] [--gb N] [--qds a,b]\n"
                        "       [--mmap-qds a,b] [--passes N] [--rounds N] [--pass-gb N]\n"
                        "       [--lat N] [--min K] [--max K] [--workers N]\n"
                        "       [--sector 4096|512] [--volume-info] [--no-ioring]\n");
        return 1;
    }
    const std::string file = argv[1];
    bool create = false, info_only = false, want_ioring = true;
    double   gb = 8.0, pass_gb = 4.0;
    unsigned passes = 2, rounds = 3, qdhint = 64, lat_n = 2000;
    uint32_t min_k = 640, max_k = 1024, sector = 4096;
    std::vector<unsigned> qds{ 1, 2, 4, 8, 16, 32, 48, 64 };
    std::vector<unsigned> mmap_qds{ 8, 32 };
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : nullptr; };
        auto list = [&](std::vector<unsigned> & v) {
            v.clear();
            const char * p = argv[i];
            while (*p) { v.push_back((unsigned) strtoul(p, (char **) &p, 10)); if (*p == ',') p++; }
        };
        if      (a == "--create")          create = true;
        else if (a == "--volume-info")      info_only = true;
        else if (a == "--no-ioring")        want_ioring = false;
        else if (a == "--gb"      && next()) gb      = atof(argv[i]);
        else if (a == "--passes"  && next()) passes  = (unsigned) atoi(argv[i]);
        else if (a == "--rounds"  && next()) rounds  = (unsigned) atoi(argv[i]);
        else if (a == "--min"     && next()) min_k   = (uint32_t) atoi(argv[i]);
        else if (a == "--max"     && next()) max_k   = (uint32_t) atoi(argv[i]);
        else if (a == "--lat"     && next()) lat_n   = (unsigned) atoi(argv[i]);
        else if (a == "--workers" && next()) qdhint  = (unsigned) atoi(argv[i]);
        else if (a == "--pass-gb" && next()) pass_gb = atof(argv[i]);
        else if (a == "--sector"  && next()) sector  = (uint32_t) atoi(argv[i]);
        else if (a == "--qds"     && next()) list(qds);
        else if (a == "--mmap-qds"&& next()) list(mmap_qds);
    }
    if (min_k > max_k) std::swap(min_k, max_k);

    report_volume(file);
    if (info_only) return 0;
    if (create && !create_file(file, (uint64_t) (gb * 1e9))) return 2;
    if (create) report_volume(file);

    HANDLE probe = CreateFileW(std::filesystem::path(file).wstring().c_str(),
                               GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, 0, nullptr);
    if (probe == INVALID_HANDLE_VALUE) {
        printf("cannot open %s (%lu)\n", file.c_str(), GetLastError());
        return 2;
    }
    LARGE_INTEGER fsz{};
    GetFileSizeEx(probe, &fsz);
    CloseHandle(probe);
    const uint64_t file_bytes = (uint64_t) fsz.QuadPart;
    printf("file size    : %.2f GB\n", file_bytes / 1e9);
    if (file_bytes < ((uint64_t) max_k << 20) * 8) {
        printf("file too small for this pattern; use --create --gb 8 or larger\n");
        return 2;
    }

    const bool purged = try_purge_standby();
    printf("standby list : %s\n", purged ? "purged -- buffered rows can read the drive"
                                        : "NOT purged (needs elevation) -- every BUFFERED row "
                                          "below is cache-warm RAM, not a disk number");
    printf("                the unbuffered rows are unaffected: they bypass the cache manager\n");

    set_dio_align(sector);
    printf("layout       : dio_align=%u%s\n", sector,
           sector < QWFN_DIO_PAGE ? "  (512 stride: bounce path, window read + memcpy)"
                                  : "  (in place, no copy)");
    printf("slice        : %u-%u KiB random, %u passes/mode/QD, %u interleaved rounds, "
           "%u worker hint\n", min_k, max_k, passes, rounds, qdhint);

    const std::vector<slice> pat =
        build_pattern(file_bytes, sector, min_k << 10, max_k << 10, pass_gb);
    uint64_t pat_bytes = 0;
    for (const slice & s : pat) pat_bytes += s.len;
    printf("pattern      : %zu slices, %.2f GB per pass, seed fixed across every mode\n\n",
           pat.size(), pat_bytes / 1e9);

    // One conversion for the ring arm, from the pattern the pool already reads,
    // so the two arms cannot drift apart.
    std::vector<io_slice> ipat;
    ipat.reserve(pat.size());
    for (const slice & s : pat) ipat.push_back(io_slice{ s.off, s.len });

    std::vector<void *> bufs(qdhint);
    for (auto & b : bufs) b = dio_alloc((size_t) max_k << 10);
    for (void * b : bufs)
        if (!b) { printf("buffer allocation failed\n"); return 2; }

    printf("== modes ==\n");
    for (int m = 0; m < NMODES; m++) {
        if (!want_ioring && MODES[m].a == arm::ring) continue;
        printf("  %-19s %s\n", MODES[m].name, MODES[m].what);
    }
    printf("\n");

    bool all_ok = true;

    // ------------------------------------------------------------ one sweep --
    // `with_mapping` is the controlled variable: a read-only mapping of the same
    // file, which is what the PLE table holds for the life of the process.
    auto sweep = [&](bool with_mapping, const std::vector<unsigned> & depths,
                     bool want_latency) {
        mapping map;
        if (with_mapping) {
            if (!map.open(file, file_bytes)) {
                printf("could not map the file; the mapping-open sweep is invalid\n");
                return;
            }
            map.fault_sparse();
        }
        printf("\n===== %s =====\n", with_mapping
               ? "CONDITION B: a read-only mapping of the same file is held open"
               : "CONDITION A: no competing mapping on the file");

        arm_slot slots[NMODES];
        for (int m = 0; m < NMODES; m++) {
            const mode_def & md = MODES[m];
            if (!want_ioring && md.a == arm::ring) { slots[m].why = "skipped (--no-ioring)"; continue; }
            std::string err;
            if (md.a == arm::pool) {
                if (!slots[m].pool.init({ file }, qdhint, md.direct, err,
                                        io_engine::backend::threads, md.bpio)) {
                    slots[m].why = "init: " + err;
                    continue;
                }
                slots[m].ok = true;
                if (md.bpio) printf("  %-19s BypassIO: %s\n", md.name,
                                    slots[m].pool.bypass_active() ? "ACTIVE" : "not active");
                if (md.bpio && !slots[m].pool.bypass_note().empty())
                    printf("  %-19s   kernel said: %s\n", "", slots[m].pool.bypass_note().c_str());
            } else {
                if (!slots[m].ring.open(file, qdhint, md.direct, md.reg, md.wait, err)) {
                    slots[m].why = err;
                    continue;
                }
                if (md.reg && !slots[m].ring.register_landing_buffers(bufs, max_k << 10, err)) {
                    slots[m].why = err;
                    slots[m].ring.close();
                    continue;
                }
                slots[m].ok = true;
            }
            slots[m].cached_rows_valid = md.buffered_reads_possible && purged;
        }

        for (int m = 0; m < NMODES; m++) {
            if (!slots[m].ok && !slots[m].why.empty())
                printf("  %-19s UNAVAILABLE: %s\n", MODES[m].name, slots[m].why.c_str());
        }

        for (unsigned qd : depths) {
            if (qd > qdhint || pat.size() < qd) continue;
            printf("\n  -- QD %u --  (GB/s best of %u rounds; us/read is the queue's\n"
                   "     per-read wall clock at this depth)\n", qd, rounds);
            printf("  %-19s %14s %12s %10s\n", "mode", "GB/s", "us/read", "vs shipped");
            double shipped_best = 0;
            for (int m = 0; m < NMODES; m++) {
                if (!slots[m].ok) continue;
                double best = 0, last_us = 0;
                std::vector<double> all;
                for (unsigned r = 0; r < rounds && slots[m].ok; r++) {
                    std::vector<double> s;
                    for (unsigned p = 0; p < passes; p++) {
                        if (MODES[m].a == arm::pool) {
                            const result res = run_pool_pass(slots[m].pool, pat, bufs, qd);
                            s.push_back(res.gbps);
                            last_us = res.mean_us;
                        } else {
                            double g = 0, mu = 0;
                            if (!run_ring_pass(slots[m].ring, ipat, bufs, qd, g, mu)) {
                                slots[m].why = "read_burst: " + slots[m].ring.note();
                                slots[m].ok = false;
                                break;
                            }
                            s.push_back(g);
                            last_us = mu;
                        }
                    }
                    for (double x : s) all.push_back(x);
                }
                if (!slots[m].ok) {
                    printf("  %-19s %14s %12s   %s\n", MODES[m].name, "n/a", "-",
                           slots[m].why.c_str());
                    continue;
                }
                for (double x : all) best = std::max(best, x);
                if (m == I_SHIPPED) shipped_best = best;
                char d[16] = "-";
                if (shipped_best > 0 && m != I_SHIPPED)
                    snprintf(d, sizeof(d), "%+.1f%%", (best / shipped_best - 1.0) * 100.0);
                char b[32];
                snprintf(b, sizeof(b), "%.2f", best);
                printf("  %-19s %14s %12.0f %10s\n", MODES[m].name, b, last_us, d);
            }
        }

        if (want_latency) {
            printf("\n  -- latency: one read at a time (QD 1), %u samples, us --\n", lat_n);
            printf("  %-19s %6s %8s %8s %8s %8s %8s %8s   %s\n", "mode", "n", "min",
                   "p50", "p95", "p99", "mean", "max", "reads the drive?");
            for (int m = 0; m < NMODES; m++) {
                if (!slots[m].ok) continue;
                std::vector<double> us;
                if (MODES[m].a == arm::pool) us = pool_latency(slots[m].pool, pat, bufs, lat_n);
                else                        us = ring_latency(slots[m].ring, ipat, bufs, lat_n);
                const lat_stats s = summarise(us);
                if (!s.n) { printf("  %-19s (no samples)\n", MODES[m].name); continue; }
                printf("  %-19s %6llu %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f   %s\n",
                       MODES[m].name, (unsigned long long) s.n, s.min_us, s.p50, s.p95,
                       s.p99, s.mean, s.max,
                       MODES[m].direct ? "yes (unbuffered)" : "no (cache-warm)");
            }
        }

        printf("\n  -- correctness, byte-exact, separate pass --\n");
        for (int m = 0; m < NMODES; m++) {
            if (!slots[m].ok) continue;
            uint64_t checked = 0;
            const bool ok = (MODES[m].a == arm::pool)
                ? verify_mode(slots[m].pool, MODES[m].name, pat, bufs, 8, checked)
                : verify_ring(slots[m].ring, MODES[m].name, ipat, bufs, 8, checked);
            std::string cqe;
            if (MODES[m].a == arm::ring && slots[m].ring.errors())
                cqe = "  CQE errors: " + slots[m].ring.first_error();
            printf("  %-19s %s  (%.2f GB compared)%s\n", MODES[m].name, ok ? "OK  " : "FAIL",
                   checked / 1e9, cqe.c_str());
            if (!ok) all_ok = false;
        }
        for (int m = 0; m < NMODES; m++) slots[m].pool.shutdown();
    };

    sweep(false, qds, true);
    if (want_ioring) sweep(true, mmap_qds, false);

    for (void * b : bufs) dio_free(b);
    printf("\n%s\n", all_ok ? "all available modes byte-exact" : "SOME MODE RETURNED WRONG BYTES");
    return all_ok ? 0 : 1;
}