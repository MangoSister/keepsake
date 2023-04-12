#include "memory_util.h"
#include "assertion.h"
#include <algorithm>
#include <cstddef>

namespace ks
{

BlockAllocator::BlockAllocator(size_t default_block_size, size_t max_num_blocks)
    : default_block_size(default_block_size), max_num_blocks(max_num_blocks)
{}

BlockAllocator::~BlockAllocator()
{
    free_aligned(curr_block);
    for (auto &block : used_blocks) {
        free_aligned(block.second);
    }
    for (auto &block : free_blocks) {
        free_aligned(block.second);
    }
}

BlockAllocator::BlockAllocator(BlockAllocator &&other) : BlockAllocator() { swap(other); }

BlockAllocator &BlockAllocator::operator=(BlockAllocator &&other)
{
    swap(other);
    return *this;
}

void BlockAllocator::swap(BlockAllocator &other)
{
    std::swap(default_block_size, other.default_block_size);
    std::swap(curr_block_pos, other.curr_block_pos);
    std::swap(curr_alloc_size, other.curr_alloc_size);
    std::swap(curr_block, other.curr_block);
    std::swap(used_blocks, other.used_blocks);
    std::swap(free_blocks, other.free_blocks);
    std::swap(max_num_blocks, other.max_num_blocks);
}

void *BlockAllocator::allocate(size_t byteCount)
{
    // constexpr size_t alignment = alignof(std::max_align_t);
    constexpr size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

    byteCount = (byteCount + alignment - 1) & ~(alignment - 1);
    if (curr_block_pos + byteCount > curr_alloc_size) {
        // Add current block to usedBlocks list.
        if (curr_block) {
            used_blocks.push_back(std::make_pair(curr_alloc_size, curr_block));
            ASSERT(used_blocks.size() + free_blocks.size() <= max_num_blocks,
                   "BlockAllocator: total allocated blocks exceeds maximum (%lu). Consider relaxing the threshold or "
                   "debugging leaks!",
                   max_num_blocks);
            curr_block = nullptr;
            curr_alloc_size = 0;
        }

        // Get a new block of memory.

        // Try to get memory block from freeBlocks.
        for (size_t i = 0; i < free_blocks.size(); ++i) {
            if (free_blocks[i].first >= byteCount) {
                curr_alloc_size = free_blocks[i].first;
                curr_block = free_blocks[i].second;
                // (swap and shrink)
                if (i + 1 < free_blocks.size()) {
                    std::swap(free_blocks[i], free_blocks.back());
                }
                free_blocks.resize(free_blocks.size() - 1);
                break;
            }
        }
        // Didn't find one free. Need to allocate a new block.
        if (!curr_block) {
            curr_alloc_size = std::max(byteCount, default_block_size);
            constexpr size_t kCacheLine = 64;
            curr_block = (byteptr_t)alloc_aligned(curr_alloc_size, kCacheLine);
        }
        curr_block_pos = 0;
    }
    void *ret = curr_block + curr_block_pos;
    curr_block_pos += byteCount;
    return ret;
}

void BlockAllocator::free(void *bytes)
{
    (void)bytes;
    // This function is a no-op.
    // PoolAllocator can only grow. It can only free all allocated memory at once (during destruction).
}

void BlockAllocator::reset()
{
    free_blocks.insert(free_blocks.end(), std::make_move_iterator(used_blocks.begin()),
                       std::make_move_iterator(used_blocks.end()));
    used_blocks.clear();
}

} // namespace ks