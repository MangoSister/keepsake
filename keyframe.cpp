#include "keyframe.h"

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
