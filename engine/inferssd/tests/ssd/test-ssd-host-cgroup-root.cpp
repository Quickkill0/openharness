// SPDX-License-Identifier: Apache-2.0
// Copyright OpenHarness contributors.
// A memory.max that does not exist means the memory controller is not enabled in that directory, so the walk carries
// on to its parent, and with none on the path no limit applies. Inside a cgroup namespace the root shown can be a
// limited cgroup, so only "no such file" counts as absent; any other read error is an unknown limit.
#include "../../ggml/src/ggml-ssd/host/ledger.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

using namespace ssd::host;

static void require(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why); } }

// A ledger for /sys/fs/cgroup/app with an unlimited leaf and a root whose memory.max fails with root_errno.
static room measured(kimi_host_ledger & ledger, int root_errno) {
    ledger.cgroup = "/sys/fs/cgroup/app";
    ledger.reserve = 512ull << 20;
    ledger.read = [root_errno](const std::string & path, std::string & text) {
        if (path == "/sys/fs/cgroup/app/memory.max") { text = "max\n"; return true; }
        if (path == "/sys/fs/cgroup/memory.max") { errno = root_errno; return false; }
        if (path == "/proc/meminfo") { text = "MemAvailable: 8388608 kB\n"; return true; }
        errno = ENOENT;
        return false;
    };
    return ledger.measure(0);
}

// The limit admission uses. A path that is not in `files` is absent.
static bool limit_of(const std::map<std::string, std::string> & files, int root_errno, uint64_t & limit,
                     const std::string & cgroup = "/sys/fs/cgroup/a/b", bool * found = nullptr) {
    kimi_host_ledger ledger;
    ledger.cgroup = cgroup;
    ledger.read = [files, root_errno](const std::string & path, std::string & text) {
        const auto it = files.find(path);
        if (it == files.end()) { errno = path == "/sys/fs/cgroup/memory.max" ? root_errno : ENOENT; return false; }
        text = it->second;
        return true;
    };
    return ledger.memory_limit(limit, found);
}

static void test_memory_limit() {
    const std::string a = "/sys/fs/cgroup/a/memory.max", b = "/sys/fs/cgroup/a/b/memory.max";
    uint64_t limit = 0;
    require(limit_of({{a, "max\n"}, {b, "max\n"}}, ENOENT, limit) && limit == UINT64_MAX, "no finite limit, absent root");
    require(limit_of({{a, "2147483648\n"}, {b, "1073741824\n"}}, ENOENT, limit) && limit == 1073741824, "the leaf is the tightest");
    require(limit_of({{a, "1073741824\n"}, {b, "max\n"}}, ENOENT, limit) && limit == 1073741824, "an ancestor limits too");
    require(limit_of({{a, "max\n"}, {b, "max\n"}, {"/sys/fs/cgroup/memory.max", "536870912\n"}}, ENOENT, limit) && limit == 536870912,
            "a namespace root can be finite");
    for (const int error : { EACCES, EIO, EPERM }) {
        require(!limit_of({{a, "max\n"}, {b, "max\n"}}, error, limit), "an unreadable root memory.max is an unknown limit");
    }
    require(!limit_of({{a, "many\n"}, {b, "max\n"}}, ENOENT, limit), "an unparseable memory.max is an unknown limit");

    // No memory controller in a directory: the nearest ancestor that has the file decides.
    const std::string leaf = "/sys/fs/cgroup/a/b";
    bool found = false;
    require(limit_of({{b, "max\n"}}, ENOENT, limit, leaf, &found) && limit == UINT64_MAX && found, "a directory without the file is passed over");
    require(limit_of({{a, "1073741824\n"}}, ENOENT, limit, leaf, &found) && limit == 1073741824 && found, "an ancestor limits a leaf without the file");
    require(limit_of({}, ENOENT, limit, leaf, &found) && limit == UINT64_MAX && !found, "no controller on the path is no limit");
    require(limit_of({}, ENOENT, limit, "", &found) && limit == UINT64_MAX && !found, "no cgroup v2 path (cgroup v1) is no limit");

    kimi_host_ledger denied;
    denied.cgroup = "/sys/fs/cgroup/a";
    denied.read = [](const std::string &, std::string &) { errno = EACCES; return false; };
    require(!denied.memory_limit(limit), "an unreadable memory.max below the root is an unknown limit");
    denied.read = [](const std::string &, std::string &) { errno = 0; return false; };
    require(!denied.memory_limit(limit), "a failed read with no reason below the root is an unknown limit");
}

// Finding no cgroup v2 path is an answer; failing to read /proc/self/cgroup is not, whatever the reason.
static void test_discovery() {
    const auto reader = [](int error, const char * groups) {
        return [error, groups](const std::string & path, std::string & text) {
            if (path == "/proc/self/cgroup") {
                if (error) { errno = error; return false; }
                text = groups;
                return true;
            }
            if (path == "/proc/meminfo") { text = "MemAvailable: 8388608 kB\n"; return true; }
            errno = ENOENT;
            return false;
        };
    };
    for (const int error : { EACCES, EIO, EPERM, ENOENT }) {
        kimi_host_ledger ledger;
        ledger.reserve = 512ull << 20;
        ledger.read = reader(error, "");
        ledger.locate();
        uint64_t limit = 0;
        require(!ledger.memory_limit(limit), "a failed read of /proc/self/cgroup is an unknown limit");
        require(!ledger.unbounded(), "the arbiter read a failed discovery as unbounded");
        const room r = ledger.measure(0);
        require(!r.envelope_unbounded && r.cgroup == room_unreadable, "a failed discovery did not fail closed");

        // Reading it again clears the failure.
        ledger.read = reader(0, "12:memory:/app\n");
        ledger.locate();
        require(ledger.unbounded(), "a later successful discovery still counted as failed");
    }
    kimi_host_ledger v1;
    v1.reserve = 512ull << 20;
    v1.read = reader(0, "12:memory:/app\n1:name=systemd:/app\n");
    v1.locate();
    uint64_t limit = 0;
    bool found = true;
    require(v1.memory_limit(limit, &found) && limit == UINT64_MAX && !found, "cgroup v1 alone is no controller, not a failure");
    require(v1.measure(0).envelope_unbounded, "cgroup v1 alone left the envelope unreadable");
}

// The arbiter's term: with no controller the RAM cache still works, held to MemAvailable.
static void test_no_controller() {
    const auto no_files = [](const std::string & path, std::string & text) {
        if (path == "/proc/meminfo") { text = "MemAvailable: 8388608 kB\n"; return true; }
        errno = ENOENT;
        return false;
    };
    for (const std::string & cgroup : { std::string("/sys/fs/cgroup/app"), std::string() }) {
        kimi_host_ledger ledger;
        ledger.cgroup = cgroup;
        ledger.reserve = 512ull << 20;
        ledger.read = no_files;
        const room r = ledger.measure(0);
        require(r.envelope_unbounded, "no memory controller left the envelope unreadable");
        require(r.limits != room_unreadable, "no memory controller left the limits unreadable");
        require(ledger.unbounded(), "no memory controller is no limit (arbiter)");
    }
}

int main() {
    try {
        test_memory_limit();
        test_discovery();
        test_no_controller();
        kimi_host_ledger absent;
        const room none = measured(absent, ENOENT);
        require(none.envelope_unbounded, "an absent root memory.max is no limit");
        require(absent.unbounded(), "an absent root memory.max is no limit (arbiter)");

        kimi_host_ledger injected; // a reader that does not set errno means the file is absent
        require(measured(injected, 0).envelope_unbounded, "a reader that sets no errno means absent");

        for (const int error : { EACCES, EIO, EPERM }) {
            kimi_host_ledger unreadable;
            const room r = measured(unreadable, error);
            require(!r.envelope_unbounded, "an unreadable root memory.max became unbounded");
            require(r.cgroup == room_unreadable, "an unreadable root memory.max did not fail closed");
            require(!unreadable.unbounded(), "the arbiter read an unreadable root memory.max as unbounded");
        }
        std::puts("ok");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
