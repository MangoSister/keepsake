#pragma once
#include <chrono>
#include <filesystem>
#include <iomanip>
namespace fs = std::filesystem;

namespace ks
{

static inline std::string current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S");
    return ss.str();
}

template <typename Func>
void test_case(const fs::path &root_dir, const std::string &test_name, const Func &test_func)
{
    printf("Start test [%s]...\n", test_name.c_str());
    fs::path output_dir = root_dir / test_name;
    if (!fs::create_directory(output_dir)) {
        printf("Cannot create output directory for [%s]. Skipping...\n", test_name.c_str());
        return;
    }

    auto start = std::chrono::system_clock::now();
    test_func(output_dir);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> duration = end - start;
    printf("Finished [%s]. Took %.3f sec.\n", test_name.c_str(), duration.count());
}

} // namespace ks