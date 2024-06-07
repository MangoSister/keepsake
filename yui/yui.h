#pragma once

#include "../assertion.h"
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#define UUID_SYSTEM_GENERATOR
#include <uuid.h>
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace ks
{
namespace yui
{

using uuid = uuids::uuid;

struct AssetBase
{
    virtual ~AssetBase() = default;
    AssetBase(uuid &&id, sol::table &&metadata) : id(std::move(id)), metadata(std::move(metadata)) {}

    uuid id;
    sol::table metadata;
};

template <typename T>
struct Asset : public AssetBase
{
    Asset(uuid &&id, std::shared_ptr<T> content)
        : AssetBase(std::move(id), content->generate_metadata()), content(content)
    {}

    // also need to pass the table itself or some kind of context
    std::shared_ptr<T> load()
    {
        std::shared_ptr<T> loaded = T::load_from_metadata(metadata);
        content = loaded;
        return loaded;
    }

    std::weak_ptr<T> content;
};

struct AssetTable;

template <typename T>
struct AssetRef
{
    void clear() { ptr = nullptr; }

    T *operator->() { return &*ptr; }
    const T *operator->() const { return &*ptr; }

    uuid id;
    std::shared_ptr<T> ptr;
};

// AssetRef<T>
// uuid id
// const T *obj = nullptr

// asset:
// can be created from lua script
// can be created from c++
// can be referenced by an (unique?) asset path
// can be nested
// hot load (?!)

// Asset has 3 states:
// non-existent
// unloaded
// loaded

struct AssetTableHash
{
    using is_transparent = void;
    size_t operator()(const uuid &id) const { return std::hash<uuid>{}(id); }
    size_t operator()(const std::unique_ptr<AssetBase> &asset) const { return std::hash<uuid>{}(asset->id); }
};

struct AssetTableEqual
{
    using is_transparent = void;
    bool operator()(const std::unique_ptr<AssetBase> &asset, const uuid &id) const { return asset->id == id; }
    bool operator()(const uuid &id, const std::unique_ptr<AssetBase> &asset) const { return asset->id == id; }
    bool operator()(const std::unique_ptr<AssetBase> &asset1, const std::unique_ptr<AssetBase> &asset2) const
    {
        return asset1->id == asset2->id;
    }
    bool operator()(const uuid &id1, const uuid &id2) const { return id1 == id2; }
};

struct AssetTable
{
    template <typename T>
    AssetRef<T> add(std::shared_ptr<T> content)
    {
        ASSERT(content);
        auto asset = std::make_unique<Asset<T>>(uuid_gen(), content);
        uuid id = asset->id;
        auto insert = table.insert(std::move(asset));
        ASSERT(insert.second, "AssetTable UUID duplication?!");
        return AssetRef<T>{id, content};
    }

    template <typename T>
    bool lookup(AssetRef<T> &ref) const
    {
        if (ref.id.is_nil()) {
            return false;
        }
        auto it = table.find(ref.id);
        if (it == table.end()) {
            return false;
        }

        Asset<T> *asset = dynamic_cast<Asset<T> *>(it->get());
        if (!asset) {
            return false;
        }
        ref.ptr = asset->content.lock();
        if (ref.ptr) {
            return true;
        }

        ref.ptr = asset->load();
        return (bool)ref.ptr;
    }

    std::unordered_set<std::unique_ptr<AssetBase>, AssetTableHash, AssetTableEqual> table;
    uuids::uuid_system_generator uuid_gen;
};

// template <typename T>
// T *AssetRef<T>::get() const
//{
//     if (std::shared_ptr<T> spt = ptr.lock()) {
//
//     } else {
//         return nullptr;
//     }
// }

// template <typename T>
// T *AssetRef<T>::acquire(const AssetTable &table)
//{
//     if (id.is_nil()) {
//         return nullptr;
//     }
//     if (asset.expired()) {
//
//     }
// }

// scriptable:
// can be created and manipulated from lua script
// hierarchy/scene graph?

// transform?

// camera
struct Scriptable
{
    // virtual void;
};

} // namespace yui

void foo();

} // namespace ks