#include "IFeatureSelector.h"
#include <thread>
#include <iostream>
using namespace std;
class ExtraTreesRandomSelector : public IFeatureSelector
{

private:
    struct MinMax
    {
        float16_t min, max;
        MinMax() : min(65504), max(0) {};
    };
    MinMax *lookUpTable;
    mutex *mut;
    bool generatedLimits;
    int n_features;
    int n_samples;
    int n_threads;

    // 2 options for getting min and max values:
    // 1. is to parallelize column wise, each thread doing one attribute. That would cause to many cache misses - every thread needs to read the whole row every time
    // 2. option is to parallelize row wise, which should be faster.
    // Another option: get the whole column, sort it, get first - no mutex locks, faster branch pred, worse cache
    void GetColumnMinMaxf(float16_t **X, int n_samples, int j);

    void GetRowMinMax(float16_t **X, int i_s, int i_e);

public:
    ExtraTreesRandomSelector(int n_features);
    ~ExtraTreesRandomSelector();

    // For every trained tree it takes a random feature, gets the min and max and then takes some random value between those two
    //  For speed up, make a look up table
    float16_t GetThreshold(float16_t **X, int *labels, int n_samples, int feature_index);
};