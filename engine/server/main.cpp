// SPDX-License-Identifier: MIT
// OpenHarness engine server entry point. Replaces llama.cpp's
// tools/server/main.cpp; the server itself is upstream's llama-server-impl.
//
// `openharness-engine-server --engine-info` prints a JSON capability record
// (engine version, upstream base, compiled backends and devices, feature
// flags) and exits. Every other invocation is handed to llama-server unchanged.

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>

int llama_server(int argc, char ** argv);

namespace {

std::string json_str(const char * s) {
    std::string out = "\"";
    for (const char * p = s ? s : ""; *p; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

const char * dev_type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        default:                             return "unknown";
    }
}

int print_engine_info() {
    ggml_backend_load_all();

    std::string out = "{";
    out += "\"engine\":\"openharness-engine\"";
    out += ",\"engine_version\":" + json_str(OH_ENGINE_VERSION);
    out += ",\"llama_cpp\":{\"tag\":" + json_str(OH_LLAMA_TAG) + ",\"commit\":" + json_str(OH_LLAMA_COMMIT) + "}";
    // Features land with later engine stages; the sidecar must not assume
    // them from the backend list (CUDA dense is not the CUDA expert tier).
    // remote_downloads follows the compile-time no-download rule (set on
    // llama-common by engine/CMakeLists.txt), so the record cannot claim it
    // while the code would still download.
#ifdef OPENHARNESS_NO_REMOTE
    const char * remote_downloads = "false";
#else
    const char * remote_downloads = "true";
#endif
    // cuda_vram_expert_tier: this build compiled the tier into its CUDA
    // backend. It says nothing about whether this machine has a usable GPU.
#ifdef OH_VRAM_TIER
    const char * vram_tier = "true";
#else
    const char * vram_tier = "false";
#endif
    // ssd_expert_streaming: --ssd works (CPU sink, RAM cache, reads from the
    // original GGUF). Which architectures it accepts is checked at load, and
    // an unsupported one fails the load with a reason.
    out += ",\"features\":{\"ssd_expert_streaming\":true,\"cuda_vram_expert_tier\":" + std::string(vram_tier) + ",\"web_ui\":false";
    out += ",\"remote_downloads\":" + std::string(remote_downloads) + "}";
    out += ",\"backends\":[";
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (i) out += ",";
        out += "{\"name\":" + json_str(ggml_backend_reg_name(reg)) + ",\"devices\":[";
        for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            if (j) out += ",";
            out += "{\"name\":" + json_str(ggml_backend_dev_name(dev));
            out += ",\"description\":" + json_str(ggml_backend_dev_description(dev));
            out += ",\"type\":" + json_str(dev_type_name(ggml_backend_dev_type(dev)));
            out += ",\"memory_total_mib\":" + std::to_string(total_b / (1024 * 1024));
            out += ",\"memory_free_mib\":" + std::to_string(free_b / (1024 * 1024)) + "}";
        }
        out += "]}";
    }
    out += "]}";
    std::printf("%s\n", out.c_str());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--engine-info") == 0) {
        return print_engine_info();
    }
    return llama_server(argc, argv);
}
