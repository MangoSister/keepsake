#include "log_util.h"
#include "assertion.h"
#include <memory>

namespace ks
{

std::unique_ptr<spdlog::logger> default_logger;

spdlog::logger &create_default_logger(const std::optional<fs::path> file_path)
{
    std::vector<spdlog::sink_ptr> sinks;
    auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[default] [%Y-%C-%d %T.%e] [%l] %v");
    sinks.push_back(console_sink);

    if (file_path) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_path->string(), true);
        file_sink->set_level(spdlog::level::info);
        file_sink->set_pattern("[default] [%Y-%C-%d %T.%e] [%l] %v");
        sinks.push_back(file_sink);
    }

    default_logger = std::make_unique<spdlog::logger>("default_ks_logger", sinks.begin(), sinks.end());
    default_logger->set_level(spdlog::level::trace);
    return *default_logger;
}

spdlog::logger &get_default_logger()
{
    ASSERT(default_logger);
    return *default_logger;
}

} // namespace ks