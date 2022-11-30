#pragma once
#include "maths.h"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
namespace fs = std::filesystem;

// heterogeneous lookup: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1690r1.html
// https://www.cppstories.com/2021/heterogeneous-access-cpp20/
struct string_hash
{
    using is_transparent = void;
    size_t operator()(const char *txt) const { return std::hash<std::string_view>{}(txt); }
    size_t operator()(std::string_view txt) const { return std::hash<std::string_view>{}(txt); }
    size_t operator()(const std::string &txt) const { return std::hash<std::string>{}(txt); }
};
template <typename T>
using StringHashTable = std::unordered_map<std::string, T, string_hash, std::equal_to<>>;

struct ConfigurableTable;
struct ConfigArgsInternal;

struct ConfigArgs
{
    ~ConfigArgs();
    ConfigArgs(std::unique_ptr<ConfigArgsInternal> &&args);
    ConfigArgs(const ConfigArgs &other);

    ConfigArgs operator[](std::string_view key) const;
    bool contains(std::string_view key) const;

    ConfigArgs operator[](int idx) const;
    size_t array_size() const;

    int load_integer(std::string_view name, const std::optional<int> &default_value = {}) const;
    float load_float(std::string_view name, const std::optional<float> &default_value = {}) const;
    vec2 load_vec2(std::string_view name, bool force_normalize = false,
                   const std::optional<vec2> &default_value = {}) const;
    vec3 load_vec3(std::string_view name, bool force_normalize = false,
                   const std::optional<vec3> &default_value = {}) const;
    vec4 load_vec4(std::string_view name, bool force_normalize = false,
                   const std::optional<vec4> &default_value = {}) const;
    Transform load_transform(std::string_view name, const std::optional<Transform> &default_value = {}) const;
    bool load_bool(std::string_view name, const std::optional<bool> &default_value = {}) const;
    std::string load_string(std::string_view name, const std::optional<std::string> &default_value = {}) const;
    fs::path load_path(std::string_view name, const std::optional<fs::path> &default_value = {}) const;

    int load_integer(int index) const;
    float load_float(int index) const;
    vec2 load_vec2(int index, bool force_normalize = false) const;
    vec3 load_vec3(int index, bool force_normalize = false) const;
    vec4 load_vec4(int index, bool force_normalize = false) const;
    Transform load_transform(int index) const;
    bool load_bool(int index) const;
    std::string load_string(int index) const;
    fs::path load_path(int index) const;

    void update_time(float t) const;

    const ConfigurableTable &asset_table() const;

    std::unique_ptr<ConfigArgsInternal> args;
};

struct ConfigServiceInternal;

struct Configurable
{
    virtual ~Configurable() = default;
};
using ConfigurableParser = std::function<std::unique_ptr<Configurable>(const ConfigArgs &args)>;

struct ConfigurableTable
{
    void register_parser(std::string_view prefix, const ConfigurableParser &parser)
    {
        parsers.push_back({std::string(prefix), parser});
    }
    template <typename T>
    void register_parser(std::string_view prefix,
                         const std::function<std::unique_ptr<T>(const ConfigArgs &args)> &parser)
    {
        parsers.insert({std::string(prefix), [](const ConfigArgs &args) { return parser(args); }});
    }

    void load(ConfigServiceInternal &service);

    const Configurable *get(std::string_view path) const
    {
        auto it = assets.find(path);
        if (it == assets.end())
            return nullptr;
        return it->second.get();
    }
    template <typename T>
    const T *get(std::string_view path) const
    {
        return dynamic_cast<const T *>(get(path));
    }

    std::unique_ptr<Configurable> create_in_place(std::string_view prefix, const ConfigArgs &args) const
    {
        auto it = std::find_if(parsers.begin(), parsers.end(), [&](const auto &p) { return p.first == prefix; });
        if (it == parsers.end())
            return nullptr;
        return (it->second)(args);
    }

    template <typename T>
    std::unique_ptr<T> create_in_place(std::string_view prefix, const ConfigArgs &args) const
    {
        std::unique_ptr<Configurable> obj = create_in_place(prefix, args);
        T *t_obj = dynamic_cast<T *>(obj.get());
        if (!t_obj)
            return nullptr;
        obj.release();
        return std::unique_ptr<T>(t_obj);
    }

    StringHashTable<std::unique_ptr<Configurable>> assets;
    std::vector<std::pair<std::string, ConfigurableParser>> parsers;
};

// (args, task_dir, task_id);
using ConfigTask = std::function<void(const ConfigArgs &args, const fs::path &, int)>;

struct ConfigService
{
    ~ConfigService();
    ConfigService();

    void parse_file(const fs::path &file_path);
    void parse(std::string_view str);

    void register_asset(std::string_view prefix, const ConfigurableParser &parser);
    void register_task(std::string_view name, const ConfigTask &task);

    void load_assets();
    const ConfigurableTable &asset_table() const;

    fs::path output_directory() const;
    void run_all_tasks() const;

    std::unique_ptr<ConfigServiceInternal> service;
};