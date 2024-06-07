#pragma once

#include "assertion.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <optional>
#include <source_location>
namespace fs = std::filesystem;

namespace ks
{

spdlog::logger &create_default_logger(const std::optional<fs::path> file_path);
spdlog::logger &get_default_logger();

// Simple debugging utility for PT-like renderer. Maybe to be improved.
struct CPURendererThreadMonitor
{
    void record_pixel(int pixel_x, int pixel_y)
    {
        this->pixel_x = pixel_x;
        this->pixel_y = pixel_y;
    }

    void record_sample_idx(int sample_idx) { this->sample_idx = sample_idx; }

    void record_bounce(int bounce) { this->bounce = bounce; }

    int pixel_x, pixel_y;
    int sample_idx;
    int bounce;
};

extern thread_local CPURendererThreadMonitor cpu_renderer_thread_monitor;

inline void thread_monitor_check(bool condition, const std::source_location &location = std::source_location::current())
{
    if (!condition) {
        get_default_logger().critical("[cpu renderer thread monitor check] Failed at [File: {} ({}:{}), in `{}`] at "
                                      "pixel ({}, {}), sample {}, bounce {}",
                                      location.file_name(), location.line(), location.column(),
                                      location.function_name(), cpu_renderer_thread_monitor.pixel_x,
                                      cpu_renderer_thread_monitor.pixel_y, cpu_renderer_thread_monitor.sample_idx,
                                      cpu_renderer_thread_monitor.bounce);
        ASSERT(false);
    }
}

} // namespace ks