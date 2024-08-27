#pragma once
#include <arm_neon.h>
class IFeatureSelector
{
public:
    virtual float16_t GetThreshold(float16_t **X, int *labels, int n_samples, int feature_index) = 0;
};