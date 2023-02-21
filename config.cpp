#include "config.h"
#include "file_util.h"
#include "keyframe.h"
#include "test_util.h"
#include <iostream>
#include <toml.hpp>
#include <unsupported/Eigen/EulerAngles>

namespace ks
{

struct ConfigServiceInternal
{
    void parse_file(const fs::path &file_path);
    void parse(std::string_view str);

    float load_float_field(const toml::node_view<const toml::node> &args, float time = 0.0f);
    vec2 load_vec2_field(const toml::node_view<const toml::node> &args, bool force_normalize, float time = 0.0f);
    vec3 load_vec3_field(const toml::node_view<const toml::node> &args, bool force_normalize, float time = 0.0f);
    vec4 load_vec4_field(const toml::node_view<const toml::node> &args, bool force_normalize, float time = 0.0f);
    Transform load_transform_field(const toml::node_view<const toml::node> &args, float time = 0.0f);

    fs::path output_directory() const;
    void run_all_tasks() const;

    toml::parse_result cfg;

    fs::path asset_root_dir;
    ConfigurableTable asset_table;

    std::unordered_map<const toml::node *, KeyframeFloat> float_fields;
    std::unordered_map<const toml::node *, KeyframeVec2> vec2_fields;
    std::unordered_map<const toml::node *, KeyframeVec3> vec3_fields;
    std::unordered_map<const toml::node *, KeyframeVec4> vec4_fields;

    std::unordered_map<std::string, ConfigTask> task_factory;
};

void ConfigServiceInternal::parse_file(const fs::path &file_path)
{
    try {
        cfg = toml::parse_file(file_path.string());
    } catch (const toml::parse_error &err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
    }
}

void ConfigServiceInternal::parse(std::string_view str)
{
    try {
        cfg = toml::parse(str);
    } catch (const toml::parse_error &err) {
        std::cerr << "Parsing failed:\n" << err << "\n";
    }
}

float ConfigServiceInternal::load_float_field(const toml::node_view<const toml::node> &args, float time)
{
    if (args.is_number()) {
        return *args.value<float>();
    } else {
        auto it = float_fields.find(args.node());
        if (it == float_fields.end()) {
            const toml::array &times = *args["times"].as_array();
            const toml::array &values = *args["values"].as_array();
            KeyframeFloat field;
            field.times.resize(times.size());
            field.values.resize(values.size());
            for (int i = 0; i < times.size(); ++i) {
                field.times[i] = *times[i].value<float>();
                field.values[i] = *values[i].value<float>();
            }
            it = float_fields.insert({args.node(), std::move(field)}).first;
        }
        return it->second.eval(time);
    }
}

vec2 ConfigServiceInternal::load_vec2_field(const toml::node_view<const toml::node> &args, bool force_normalize,
                                            float time)
{
    if (args.is_array()) {
        const toml::array &components = *args.as_array();
        vec2 v;
        for (int i = 0; i < 2; ++i)
            v[i] = *components[i].value<float>();
        if (force_normalize)
            v.normalize();
        return v;
    } else {
        auto it = vec2_fields.find(args.node());
        if (it == vec2_fields.end()) {
            const toml::array &times = *args["times"].as_array();
            const toml::array &values = *args["values"].as_array();
            KeyframeVec2 field;
            field.times.resize(times.size());
            field.values.resize(values.size());
            for (int i = 0; i < times.size(); ++i) {
                field.times[i] = *times[i].value<float>();
                const toml::array &value = *values[i].as_array();
                for (int j = 0; j < 3; ++j)
                    field.values[i][j] = *value[j].value<float>();
                if (force_normalize)
                    field.values[i].normalize();
            }
            it = vec2_fields.insert({args.node(), std::move(field)}).first;
        }
        return it->second.eval(time);
    }
}

vec3 ConfigServiceInternal::load_vec3_field(const toml::node_view<const toml::node> &args, bool force_normalize,
                                            float time)
{
    if (args.is_array()) {
        const toml::array &components = *args.as_array();
        vec3 v;
        for (int i = 0; i < 3; ++i)
            v[i] = *components[i].value<float>();
        if (force_normalize)
            v.normalize();
        return v;
    } else {
        auto it = vec3_fields.find(args.node());
        if (it == vec3_fields.end()) {
            const toml::array &times = *args["times"].as_array();
            const toml::array &values = *args["values"].as_array();
            KeyframeVec3 field;
            field.times.resize(times.size());
            field.values.resize(values.size());
            for (int i = 0; i < times.size(); ++i) {
                field.times[i] = *times[i].value<float>();
                const toml::array &value = *values[i].as_array();
                for (int j = 0; j < 3; ++j)
                    field.values[i][j] = *value[j].value<float>();
                if (force_normalize)
                    field.values[i].normalize();
            }
            it = vec3_fields.insert({args.node(), std::move(field)}).first;
        }
        return it->second.eval(time);
    }
}

vec4 ConfigServiceInternal::load_vec4_field(const toml::node_view<const toml::node> &args, bool force_normalize,
                                            float time /*= 0.0f*/)
{
    if (args.is_array()) {
        const toml::array &components = *args.as_array();
        vec4 v;
        for (int i = 0; i < 4; ++i)
            v[i] = *components[i].value<float>();
        if (force_normalize)
            v.normalize();
        return v;
    } else {
        auto it = vec4_fields.find(args.node());
        if (it == vec4_fields.end()) {
            const toml::array &times = *args["times"].as_array();
            const toml::array &values = *args["values"].as_array();
            KeyframeVec4 field;
            field.times.resize(times.size());
            field.values.resize(values.size());
            for (int i = 0; i < times.size(); ++i) {
                field.times[i] = *times[i].value<float>();
                const toml::array &value = *values[i].as_array();
                for (int j = 0; j < 4; ++j)
                    field.values[i][j] = *value[j].value<float>();
                if (force_normalize)
                    field.values[i].normalize();
            }
            it = vec4_fields.insert({args.node(), std::move(field)}).first;
        }
        return it->second.eval(time);
    }
}

Transform ConfigServiceInternal::load_transform_field(const toml::node_view<const toml::node> &args, float time)
{
    // TODO: keyframe this

    const toml::table &table = *args.as_table();

    Transform transform;
    vec3 scale = vec3::Ones();
    if (table.contains("scale")) {
        const auto &s = *table["scale"].as_array();
        for (int i = 0; i < 3; ++i)
            scale[i] = *s[i].value<float>();
    }
    quat rotation = quat::Identity();
    if (table.contains("rotation")) {
        const auto &r = *table["rotation"].as_table();
        if (r.contains("euler")) {
            const auto &euler = *r["euler"].as_array();
            float angle0 = to_radian(*euler[0].value<float>());
            float angle1 = to_radian(*euler[1].value<float>());
            float angle2 = to_radian(*euler[2].value<float>());
            rotation = (quat)Eigen::EulerAnglesXYZf(angle0, angle1, angle2);
        } else if (r.contains("quat")) {
            const auto &quaternion = *r["quat"].as_array();
            float w = *quaternion[0].value<float>();
            float x = *quaternion[1].value<float>();
            float y = *quaternion[2].value<float>();
            float z = *quaternion[3].value<float>();
            rotation = quat(w, x, y, z).normalized();
        } else {
            ASSERT(false, "Must specify rotation as (XYZ) euler angles or quaternion.");
        }
    }
    vec3 translation = vec3::Zero();
    if (table.contains("translation")) {
        const auto &t = *table["translation"].as_array();
        for (int i = 0; i < 3; ++i)
            translation[i] = *t[i].value<float>();
    }
    return Transform(scale_rotate_translate(scale, rotation, translation));
}

fs::path ConfigServiceInternal::output_directory() const
{
    if (cfg.contains("output_dir")) {
        return fs::path(*cfg["output_dir"].value<std::string>());
    } else {
        return fs::path(current_time_and_date());
    }
}

void ConfigServiceInternal::run_all_tasks() const
{
    fs::path output_dir = output_directory();
    if (!fs::is_directory(output_dir) || !fs::exists(output_dir)) {
        if (!fs::create_directory(output_dir)) {
            printf("Failed to create directory %s\n", output_dir.string().c_str());
            return;
        }
        printf("Created output directory [%s]\n", output_dir.string().c_str());
    }
    if (!cfg.contains("task")) {
        printf("No task to run\n");
        return;
    }
    const toml::array &task_array = *cfg["task"].as_array();
    for (int task_id = 0; task_id < (int)task_array.size(); ++task_id) {
        fs::path task_dir;
        toml::table task_table = *task_array[task_id].as_table();
        if (task_table.contains("task_dir")) {
            task_dir = fs::path(*task_table["task_dir"].value<std::string>());
        } else {
            task_dir = string_format("task_%d", task_id);
        }
        task_dir = output_dir / task_dir;

        if (!fs::is_directory(task_dir) || !fs::exists(task_dir)) {
            if (!fs::create_directory(task_dir)) {
                printf("Failed to create directory %s\n", task_dir.string().c_str());
                return;
            }
        }

        if (task_table.contains("override")) {
            int base_task_id = *task_table["override"].value<int>();
            ASSERT(base_task_id < task_id);
            const toml::table &base_task_table = *task_array[base_task_id].as_table();
            ASSERT(!base_task_table.contains("override"));
            toml::table override_table = base_task_table;
            task_table.for_each([&](const toml::key &key, auto &&val) {
                if (key != "override")
                    override_table.insert_or_assign(key, val);
            });
            task_table = std::move(override_table);
        }

        printf("Next task: \n");
        std::cout << task_table << "\n" << std::endl;
        std::ofstream write_config(task_dir / "config.toml");
        write_config << task_table << std::endl;

        std::string type = *task_table["type"].value<std::string>();
        const ConfigTask &task = task_factory.at(type);

        toml::node_view<const toml::node> view(task_table);
        ConfigArgs args(std::make_unique<ConfigArgsInternal>(const_cast<ConfigServiceInternal *>(this), view));
        task(args, task_dir, task_id);

        printf("Saving output to [%s]\n\n", task_dir.string().c_str());
    }
}

void ConfigurableTable::load(ConfigServiceInternal &service)
{
    for (const auto &[field, parser] : parsers) {
        if (service.cfg.contains(field)) {
            const toml::table &table = *service.cfg[field].as_table();
            table.for_each([&](const toml::key &key, const toml::table &val) {
                toml::node_view<const toml::node> view(val);
                ConfigArgs args(std::make_unique<ConfigArgsInternal>(&service, view));
                auto asset = parser(args);
                std::string path = field + "." + std::string(key.str());
                assets.insert({std::move(path), std::move(asset)});
            });
        }
    }
}

ConfigService::~ConfigService() = default;
ConfigService::ConfigService() : service(std::make_unique<ConfigServiceInternal>()) {}

void ConfigService::parse_file(const fs::path &file_path) { service->parse_file(file_path); }

void ConfigService::parse(std::string_view str) { service->parse(str); }

void ConfigService::set_asset_root_dir(const fs::path &root) { service->asset_root_dir = fs::absolute(root); }

void ConfigService::register_asset(std::string_view prefix, const ConfigurableParser &parser)
{
    service->asset_table.register_parser(prefix, parser);
}

void ConfigService::register_task(std::string_view name, const ConfigTask &task)
{
    service->task_factory.insert({std::string(name), task});
}

void ConfigService::load_assets() { service->asset_table.load(*service); }

const ConfigurableTable &ConfigService::asset_table() const { return service->asset_table; }

fs::path ConfigService::output_directory() const { return service->output_directory(); }

void ConfigService::run_all_tasks() const { return service->run_all_tasks(); }

struct ConfigArgsInternal
{
    ConfigArgsInternal(ConfigServiceInternal *service, toml::node_view<const toml::node> args)
        : service(service), args(args)
    {}

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
    fs::path load_path(std::string_view name) const;

    int load_integer(int index) const;
    float load_float(int index) const;
    vec2 load_vec2(int index, bool force_normalize = false) const;
    vec3 load_vec3(int index, bool force_normalize = false) const;
    vec4 load_vec4(int index, bool force_normalize = false) const;
    Transform load_transform(int index) const;
    bool load_bool(int index) const;
    std::string load_string(int index) const;
    fs::path load_path(int index) const;

    ConfigServiceInternal *service;
    toml::node_view<const toml::node> args;
    mutable float time = 0.0f;
};

int ConfigArgsInternal::load_integer(std::string_view name, const std::optional<int> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    const auto &v = args[name].value<int>();
    if (v)
        return *v;
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No integer value named [%.*s].", static_cast<int>(name.length()), name.data());
        return 0;
    }
}

int ConfigArgsInternal::load_integer(int index) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    const auto &v = args[index].value<int>();
    if (v)
        return *v;
    else {
        ASSERT(false, "No integer value at [%d].", index);
        return 0;
    }
}

float ConfigArgsInternal::load_float(std::string_view name, const std::optional<float> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return service->load_float_field(args[name], time);
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No float value named [%.*s].", static_cast<int>(name.length()), name.data());
        return 0.0f;
    }
}

float ConfigArgsInternal::load_float(int index) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return service->load_float_field(args[index], time);
}

vec2 ConfigArgsInternal::load_vec2(std::string_view name, bool force_normalize,
                                   const std::optional<vec2> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return service->load_vec2_field(args[name], force_normalize, time);
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No vec2 value named [%.*s].", static_cast<int>(name.length()), name.data());
        return vec2::Zero();
    }
}

vec2 ConfigArgsInternal::load_vec2(int index, bool force_normalize) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return service->load_vec2_field(args[index], force_normalize, time);
}

vec3 ConfigArgsInternal::load_vec3(std::string_view name, bool force_normalize,
                                   const std::optional<vec3> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return service->load_vec3_field(args[name], force_normalize, time);
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No vec3 value named [%.*s].", static_cast<int>(name.length()), name.data());
        return vec3::Zero();
    }
}

vec3 ConfigArgsInternal::load_vec3(int index, bool force_normalize) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return service->load_vec3_field(args[index], force_normalize, time);
}

vec4 ConfigArgsInternal::load_vec4(std::string_view name, bool force_normalize,
                                   const std::optional<vec4> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return service->load_vec4_field(args[name], force_normalize, time);
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No vec4 value named [%.*s].", static_cast<int>(name.length()), name.data());
        return vec4::Zero();
    }
}

vec4 ConfigArgsInternal::load_vec4(int index, bool force_normalize) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return service->load_vec4_field(args[index], force_normalize, time);
}

Transform ConfigArgsInternal::load_transform(std::string_view name, const std::optional<Transform> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return service->load_transform_field(args[name], time);
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No transform value named [%.*s].", static_cast<int>(name.length()), name.data());
        return Transform();
    }
}

Transform ConfigArgsInternal::load_transform(int index) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return service->load_transform_field(args[index], time);
}

bool ConfigArgsInternal::load_bool(std::string_view name, const std::optional<bool> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return *args[name].value<bool>();
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No bool value named [%.*s].", static_cast<int>(name.length()), name.data());
        return false;
    }
}

bool ConfigArgsInternal::load_bool(int index) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return *args[index].value<bool>();
}

std::string ConfigArgsInternal::load_string(std::string_view name,
                                            const std::optional<std::string> &default_value) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    if (args.as_table()->contains(name))
        return *args[name].value<std::string>();
    else if (default_value)
        return *default_value;
    else {
        ASSERT(false, "No string value named [%.*s].", static_cast<int>(name.length()), name.data());
        return std::string();
    }
}

std::string ConfigArgsInternal::load_string(int index) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    return *args[index].value<std::string>();
}

fs::path ConfigArgsInternal::load_path(std::string_view name) const
{
    ASSERT(args.is_table(), "This ConfigArgs is not a table.");
    ASSERT(args.as_table()->contains(name), "No path value named [%.*s].", static_cast<int>(name.length()),
           name.data());
    fs::path p(*args[name].value<std::string>());
    if (p.is_absolute()) {
        return p;
    } else if (!service->asset_root_dir.empty()) {
        return service->asset_root_dir / p;
    } else {
        return p;
    }
}

fs::path ConfigArgsInternal::load_path(int index) const
{
    ASSERT(args.is_array() || args.is_array_of_tables(), "This ConfigArgs is not an array.");
    ASSERT(index < args.as_array()->size(), "Index out of bound.");
    fs::path p(*args[index].value<std::string>());
    if (p.is_absolute()) {
        return p;
    } else if (!service->asset_root_dir.empty()) {
        return service->asset_root_dir / p;
    } else {
        return p;
    }
}

ConfigArgs::~ConfigArgs() = default;

ConfigArgs::ConfigArgs(std::unique_ptr<ConfigArgsInternal> &&args) : args(std::move(args)) {}

ConfigArgs::ConfigArgs(const ConfigArgs &other) : args(std::make_unique<ConfigArgsInternal>(*other.args)) {}

ConfigArgs ConfigArgs::operator[](std::string_view key) const
{
    ASSERT(args->args.is_table(), "This ConfigArgs is not a table.");
    toml::node_view<const toml::node> view = args->args[key];

    ConfigArgs child(std::make_unique<ConfigArgsInternal>(args->service, view));
    child.args->time = args->time;
    return child;
}

ConfigArgs ConfigArgs::operator[](int idx) const
{
    ASSERT(args->args.is_array() || args->args.is_array_of_tables(), "This ConfigArgs is not an array.");
    toml::node_view<const toml::node> view = args->args[idx];

    ConfigArgs child(std::make_unique<ConfigArgsInternal>(args->service, view));
    child.args->time = args->time;
    return child;
}

size_t ConfigArgs::array_size() const
{
    ASSERT(args->args.is_array() || args->args.is_array_of_tables(), "This ConfigArgs is not an array.");
    return args->args.as_array()->size();
}

bool ConfigArgs::contains(std::string_view key) const
{
    ASSERT(args->args.is_table(), "This ConfigArgs is not a table.");
    return args->args.as_table()->contains(key);
}

int ConfigArgs::load_integer(std::string_view name, const std::optional<int> &default_value) const
{
    return args->load_integer(name, default_value);
}

int ConfigArgs::load_integer(int index) const { return args->load_integer(index); }

float ConfigArgs::load_float(std::string_view name, const std::optional<float> &default_value) const
{
    return args->load_float(name, default_value);
}

float ConfigArgs::load_float(int index) const { return args->load_float(index); }

vec2 ConfigArgs::load_vec2(std::string_view name, bool force_normalize, const std::optional<vec2> &default_value) const
{
    return args->load_vec2(name, force_normalize, default_value);
}

vec2 ConfigArgs::load_vec2(int index, bool force_normalize) const { return args->load_vec2(index, force_normalize); }

vec3 ConfigArgs::load_vec3(std::string_view name, bool force_normalize, const std::optional<vec3> &default_value) const
{
    return args->load_vec3(name, force_normalize, default_value);
}

vec3 ConfigArgs::load_vec3(int index, bool force_normalize) const { return args->load_vec3(index, force_normalize); }

vec4 ConfigArgs::load_vec4(std::string_view name, bool force_normalize, const std::optional<vec4> &default_value) const
{
    return args->load_vec4(name, force_normalize, default_value);
}

vec4 ConfigArgs::load_vec4(int index, bool force_normalize) const { return args->load_vec4(index, force_normalize); }

Transform ConfigArgs::load_transform(std::string_view name, const std::optional<Transform> &default_value) const
{
    return args->load_transform(name, default_value);
}

Transform ConfigArgs::load_transform(int index) const { return args->load_transform(index); }

bool ConfigArgs::load_bool(std::string_view name, const std::optional<bool> &default_value) const
{
    return args->load_bool(name, default_value);
}

bool ConfigArgs::load_bool(int index) const { return args->load_bool(index); }

std::string ConfigArgs::load_string(std::string_view name, const std::optional<std::string> &default_value) const
{
    return args->load_string(name, default_value);
}

std::string ConfigArgs::load_string(int index) const { return args->load_string(index); }

fs::path ConfigArgs::load_path(std::string_view name) const { return args->load_path(name); }

fs::path ConfigArgs::load_path(int index) const { return args->load_path(index); }

void ConfigArgs::update_time(float t) const { args->time = t; }

const ConfigurableTable &ConfigArgs::asset_table() const { return args->service->asset_table; }

} // namespace ks