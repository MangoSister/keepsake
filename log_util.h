#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
namespace fs = std::filesystem;
#include <optional>

namespace ks
{

spdlog::logger &create_default_logger(const std::optional<fs::path> file_path);
spdlog::logger &get_default_logger();

} // namespace ks