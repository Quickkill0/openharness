// Slots sized per layer: how many a byte budget holds, what they cost and where they sit. No GPU needed.

#include "ggml-cuda/ssd/vram-slot-table.h"

#include <cassert>
#include <cstdint>
#include <vector>

static vram_slot_table make_table(size_t capacity, const std::vector<int> & priority) {
    return vram_slot_table(capacity, priority, true, 0.0, false, 16, 0, 44, 288);
}

// One token asking one layer for these experts: the slots that hit, -1 for each miss. Misses that were given a slot are
// in table.admissions, and are published so that they can hit later.
static std::vector<int> serve(vram_slot_table & table, int layer, const std::vector<int> & experts) {
    table.next_token();
    std::vector<kimi_vram_request> requests(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) {
        requests[i].layer = layer;
        requests[i].expert = experts[i];
        requests[i].weight = 1;
    }
    table.prepare(requests.data(), requests.size());
    for (size_t i = 0; i < experts.size(); ++i) {
        if (table.admissions[i] >= 0) {
            table.publish(i);
        }
    }
    return table.hits;
}

static void finish_all(vram_slot_table & table) { table.finish(); }

int main() {
    const std::vector<uint64_t> bytes = {100, 300, 200};
    const std::vector<int> priority = {4, 1, 7};

    // One round of the layers costs 600. 1000 buys one round, and 400 left over buys the first layer (100) and the
    // second (300), exactly: the third's extra slot would need 200 more.
    assert(vram_layer_slot_capacity(1000, bytes) == 5);
    assert(vram_layer_slot_bytes(5, bytes) == 1000);

    // The budget is never exceeded, and one more slot never fits.
    for (uint64_t budget = 0; budget < 4000; budget += 37) {
        const size_t capacity = vram_layer_slot_capacity(budget, bytes);
        assert(vram_layer_slot_bytes(capacity, bytes) <= budget);
        assert(vram_layer_slot_bytes(capacity + 1, bytes) > budget);
    }

    // Below one slot per layer the capacity says so: the tier refuses on it.
    assert(vram_layer_slot_capacity(599, bytes) < bytes.size());
    assert(vram_layer_slot_capacity(600, bytes) == bytes.size());

    // Each slot is as wide as its own layer and they do not overlap. The table's layer order matches the bytes' order.
    const size_t capacity = 8; // 3, 3, 2 slots
    const auto table = make_table(capacity, priority);
    const auto offsets = vram_layer_slot_offsets(table, priority, bytes);
    assert(offsets.size() == capacity + 1);
    assert(offsets.back() == vram_layer_slot_bytes(capacity, bytes));
    assert(table.layers[4].begin == 0 && table.layers[4].count == 3);
    assert(table.layers[1].begin == 3 && table.layers[1].count == 3);
    assert(table.layers[7].begin == 6 && table.layers[7].count == 2);
    for (size_t slot = 0; slot < capacity; ++slot) {
        const uint64_t expected = slot < 3 ? 100 : slot < 6 ? 300 : 200;
        assert(offsets[slot + 1] - offsets[slot] == expected);
    }

    // A wrong layer list is refused rather than laid out.
    bool refused = false;
    try {
        vram_layer_slot_offsets(table, {4, 1}, {100, 300});
    } catch (const std::exception &) {
        refused = true;
    }
    assert(refused);

    // Admission and eviction stay inside a layer's own slots.
    auto admit = make_table(capacity, priority);
    assert(serve(admit, 4, {10}) == std::vector<int>{-1}); // a miss; layer 4 owns slots 0..2
    const int layer4_slot = admit.admissions[0];
    assert(layer4_slot >= 0 && layer4_slot < 3);
    finish_all(admit);
    assert(serve(admit, 4, {10}) == std::vector<int>{layer4_slot}); // now a hit
    finish_all(admit);

    // Layer 7 owns two slots: experts 1 and 2 fill them.
    assert(serve(admit, 7, {1, 2}) == (std::vector<int>{-1, -1}));
    const int slot_of_1 = admit.admissions[0], slot_of_2 = admit.admissions[1];
    assert(slot_of_1 >= 6 && slot_of_1 < 8 && slot_of_2 >= 6 && slot_of_2 < 8 && slot_of_1 != slot_of_2);
    finish_all(admit);
    // Expert 1 is used again, so 2 is the colder one, and a new expert takes 2's slot.
    assert(serve(admit, 7, {1}) == std::vector<int>{slot_of_1});
    finish_all(admit);
    assert(serve(admit, 7, {3}) == std::vector<int>{-1});
    assert(admit.admissions[0] == slot_of_2);
    finish_all(admit);
    assert(admit.evictions == 1);

    // Layer 4's expert was not touched by layer 7's eviction.
    assert(serve(admit, 4, {10}) == std::vector<int>{layer4_slot});
    finish_all(admit);
    return 0;
}
