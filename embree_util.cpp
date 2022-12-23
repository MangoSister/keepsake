#include "embree_util.h"

KS_NAMESPACE_BEGIN

void handle_embree_error(void *user_ptr, RTCError code, const char *str)
{
    if (code == RTC_ERROR_NONE)
        return;

    printf("Embree: ");
    switch (code) {
    case RTC_ERROR_UNKNOWN:
        printf("RTC_ERROR_UNKNOWN");
        break;
    case RTC_ERROR_INVALID_ARGUMENT:
        printf("RTC_ERROR_INVALID_ARGUMENT");
        break;
    case RTC_ERROR_INVALID_OPERATION:
        printf("RTC_ERROR_INVALID_OPERATION");
        break;
    case RTC_ERROR_OUT_OF_MEMORY:
        printf("RTC_ERROR_OUT_OF_MEMORY");
        break;
    case RTC_ERROR_UNSUPPORTED_CPU:
        printf("RTC_ERROR_UNSUPPORTED_CPU");
        break;
    case RTC_ERROR_CANCELLED:
        printf("RTC_ERROR_CANCELLED");
        break;
    default:
        printf("invalid error code");
        break;
    }
    if (str) {
        printf(" (");
        while (*str)
            putchar(*str++);
        printf(")\n");
    }

    ASSERT(false);
    exit(1);
}

EmbreeDevice::EmbreeDevice(const std::string &device_config) : device(rtcNewDevice(device_config.c_str()))
{
    handle_embree_error(nullptr, rtcGetDeviceError(device), nullptr);
    rtcSetDeviceErrorFunction(device, handle_embree_error, nullptr);
    // TODO: Want this for cylinders, but my custom build embree crashes with when creating a lot of bezier curves
    // For now use a intersection/occlusion filter.

    // ssize_t culling = rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
    // ASSERT(culling, "We should enable culling for tracing curves");
    //// This only works for RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE, not the capped one.
    // ssize_t culling_curves = rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_CURVES_ENABLED);
    // ASSERT(culling_curves, "We should enable culling for tracing curves");
}

EmbreeDevice::~EmbreeDevice() { rtcReleaseDevice(device); }

KS_NAMESPACE_END