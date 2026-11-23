// OpenHarness: the grouped-prefill engine's entry point, reporting that it is not built.
//
// InferSSD's grouped prefill (group-compute-adapter.cu and group-matmul.cu) needs cuBLAS and is not part of the first
// VRAM tier. The tier asks for this table when it initializes and treats a create that returns null as "grouped prefill
// unavailable", which keeps prefill on the CPU. Decode on the GPU does not use this table.

#include "ggml-cuda/ssd/vram-compute-api.h"

#include <cstdio>

namespace {

void * stub_create(int, int, int, int, const int32_t *, const int32_t *, char * error, size_t error_size) {
    if (error && error_size) {
        std::snprintf(error, error_size, "grouped prefill is not built");
    }
    return nullptr;
}

void stub_destroy(void *) {}

int stub_reserve(void *, int, int, char *, size_t) { return -1; }
int stub_release(void *, char *, size_t) { return -1; }
float * stub_input_rows(void *) { return nullptr; }

int stub_begin_layer(void *, int, int, const struct vram_group_route *, const int32_t *, int,
                     const struct vram_group_activation *, char *, size_t) {
    return -1;
}

int stub_compute(void *, int, const struct vram_compute_expert *, struct CUevent_st *, struct CUevent_st **, char *,
                 size_t) {
    return -1;
}

int stub_finish_layer(void *, const float **, struct vram_group_layer_stats *, double *, char *, size_t) { return -1; }
void stub_geometry(void *, struct vram_group_geometry *) {}
size_t stub_bytes(void *) { return 0; }

const vram_group_compute_api stub_api = {
    3,
    sizeof(vram_group_compute_api),
    stub_create,
    stub_destroy,
    stub_reserve,
    stub_release,
    stub_input_rows,
    stub_begin_layer,
    stub_compute,
    stub_finish_layer,
    stub_geometry,
    stub_bytes,
    stub_bytes,
};

} // namespace

extern "C" const vram_group_compute_api * kimi_vram_group_compute_get_api() { return &stub_api; }
