#include "keyframe.h"

namespace ks
{

float KeyframeFloat::eval(float t) const
{
    int left, right;
    float weight;
    lerp_time(t, left, right, weight);

    return std::lerp(values[left], values[right], weight);
}

vec2 KeyframeVec2::eval(float t) const
{
    int left, right;
    float weight;
    lerp_time(t, left, right, weight);
    return lerp(values[left], values[right], weight);
}

vec3 KeyframeVec3::eval(float t) const
{
    int left, right;
    float weight;
    lerp_time(t, left, right, weight);

    if (use_slerp) {
        return slerp(values[left], values[right], weight);
    } else {
        return lerp(values[left], values[right], weight);
    }
}

vec4 KeyframeVec4::eval(float t) const
{
    int left, right;
    float weight;
    lerp_time(t, left, right, weight);

    if (use_slerp) {
        quat q1(values[left].w(), values[left].x(), values[left].y(), values[left].z());
        quat q2(values[right].w(), values[right].x(), values[right].y(), values[right].z());
        quat q = q1.slerp(t, q2);
        return vec4(q.x(), q.y(), q.z(), q.w());
    } else {
        return lerp(values[left], values[right], weight);
    }
}

} // namespace ks