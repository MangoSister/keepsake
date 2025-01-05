#pragma once
#include "ksvk.h"
#include <optional>
#include <utility>
#include <vector>

namespace ks
{

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class HashTable
{
  public:
    struct TableEntry
    {
        TableEntry() = default;
        TableEntry(const Key &key, const Value &value) : key(key), value(value), set(1) {}
        bool has_value() const { return set; }

        // TODO: handle uninitialized memory, placement new, etc.
        // https://stackoverflow.com/questions/71828288/why-is-stdaligned-storage-to-be-deprecated-in-c23-and-what-to-use-instead
        Key key;
        Value value;
        uint32_t set = 0; // Use uint32_t to be consistent with shader...
    };
    class Iterator
    {
      public:
        Iterator &operator++()
        {
            while (++ptr < end && !ptr->has_value())
                ;
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator old = *this;
            operator++();
            return old;
        }

        bool operator==(const Iterator &iter) const { return ptr == iter.ptr; }

        bool operator!=(const Iterator &iter) const { return ptr != iter.ptr; }

        Key &key() { return ptr->key; }
        const Key &key() const { return ptr->key; }

        Value &value() { return ptr->value; }
        const Value &value() const { return ptr->value; }

      private:
        friend class HashTable;
        Iterator(TableEntry *ptr, TableEntry *end) : ptr(ptr), end(end) {}
        TableEntry *ptr;
        TableEntry *end;
    };

    using iterator = Iterator;
    using const_iterator = const iterator;

    size_t size() const { return n_stored; }

    size_t capacity() const { return table.size(); }
    void clear()
    {
        table.clear();
        n_stored = 0;
    }

    HashTable() : table(8) {}

    HashTable(const HashTable &) = delete;
    HashTable &operator=(const HashTable &) = delete;

    void insert(const Key &key, const Value &value)
    {
        size_t offset = find_offset(key);
        if (table[offset].has_value() == false) {
            // grow hash table if it is too full
            if (3 * ++n_stored > capacity()) {
                grow();
                offset = find_offset(key);
            }
        }
        table[offset] = TableEntry(key, value);
    }

    bool contains(const Key &key) const { return table[find_offset(key)].has_value(); }

    const Value &operator[](const Key &key) const
    {
        size_t offset = find_offset(key);
        ASSERT(table[offset].has_value());
        return table[offset].value;
    }

    Value &operator[](const Key &key)
    {
        size_t offset = find_offset(key);
        ASSERT(table[offset].has_value());
        return table[offset].value;
    }

    const Value *try_get(const Key &key) const
    {
        size_t offset = find_offset(key);
        if (!table[offset].has_value()) {
            return nullptr;
        }
        return &table[offset].value;
    }

    Value *try_get(const Key &key)
    {
        size_t offset = find_offset(key);
        if (!table[offset].has_value()) {
            return nullptr;
        }
        return &table[offset].value;
    }

    iterator begin()
    {
        Iterator iter(table.data(), table.data() + capacity());
        while (iter.ptr < iter.end && !iter.ptr->has_value())
            ++iter.ptr;
        return iter;
    }

    iterator end() { return Iterator(table.data() + capacity(), table.data() + capacity()); }

    // TODO: make this better....
    template <typename TableEntryCopyFn>
    vk::AutoRelease<vk::Buffer> create_gpu(std::shared_ptr<vk::Allocator> allocator, VmaMemoryUsage usage,
                                           VmaAllocationCreateFlags flags, size_t gpu_table_entry_byte_size,
                                           const TableEntryCopyFn &copy_fn) const
    {
        std::vector<std::byte> copy(gpu_table_entry_byte_size * capacity());
        for (int i = 0; i < capacity(); ++i) {
            copy_fn(table[i].key, table[i].value, table[i].set, copy.data() + i * gpu_table_entry_byte_size);
        }
        return vk::AutoRelease<vk::Buffer>(allocator,
                                           VkBufferCreateInfo{
                                               .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                               .size = copy.size(),
                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           },
                                           usage, flags, copy);
    }

  private:
    size_t find_offset(const Key &key) const
    {
        size_t base_offset = Hash()(key) & (capacity() - 1);
        for (int n_probes = 0;; ++n_probes) {
            // Find offset for _key_ using quadratic probing
            size_t offset = (base_offset + n_probes / 2 + n_probes * n_probes / 2) & (capacity() - 1);
            if (table[offset].has_value() == false || KeyEqual()(key, table[offset].key))
                return offset;
        }
    }

    void grow()
    {
        size_t current_capacity = capacity();
        std::vector<TableEntry> new_table(std::max<size_t>(64, 2 * current_capacity), table.get_allocator());
        size_t newCapacity = new_table.size();
        for (size_t i = 0; i < current_capacity; ++i) {
            // insert _table[i]_ into _new_table_ if it is set
            if (!table[i].has_value())
                continue;
            size_t base_offset = Hash()(table[i].key) & (newCapacity - 1);
            for (int n_probes = 0;; ++n_probes) {
                size_t offset = (base_offset + n_probes / 2 + n_probes * n_probes / 2) & (newCapacity - 1);
                if (!new_table[offset].has_value()) {
                    new_table[offset] = std::move(table[i]);
                    break;
                }
            }
        }
        table = std::move(new_table);
    }

  public:
    std::vector<TableEntry> table;
    size_t n_stored = 0;
};

} // namespace ks