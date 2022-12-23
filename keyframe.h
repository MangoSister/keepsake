#pragma once
#include "assertion.h"
#include "maths.h"
#include <vector>

KS_NAMESPACE_BEGIN

template <typename T>
struct KeyframeField
{
    virtual ~KeyframeField() = default;
    virtual T eval(float t) const = 0;
    void lerp_time(float t, int &left, int &right, float &weight) const;

    std::vector<float> times;
    std::vector<T> values;
};

template <typename T>
void KeyframeField<T>::lerp_time(float t, int &left, int &right, float &weight) const
{
    ASSERT(!times.empty());
    t = std::clamp(t, 0.0f, 1.0f);
    if (t <= times[0]) {
        left = right = 0;
        weight = 0.0f;
        return;
    } else if (t >= times.back()) {
        left = right = (int)times.size() - 1;
        weight = 0.0f;
        return;
    }
    auto it = std::upper_bound(times.begin(), times.end(), t);
    left = (int)std::distance(times.begin(), std::prev(it));
    right = left + 1;
    weight = (t - times[left]) / (times[right] - times[left]);
}

struct KeyframeFloat : public KeyframeField<float>
{
    float eval(float t) const;
};

struct KeyframeVec2 : public KeyframeField<vec2>
{
    vec2 eval(float t) const;
};

struct KeyframeVec3 : public KeyframeField<vec3>
{
    vec3 eval(float t) const;

    bool use_slerp = false;
};

struct KeyframeVec4 : public KeyframeField<vec4>
{
    vec4 eval(float t) const;

    bool use_slerp = false;
};

KS_NAMESPACE_END