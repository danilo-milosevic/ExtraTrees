#include "ExtraTreesRandomSelector.h"
void ExtraTreesRandomSelector::GetColumnMinMaxf(float16_t **X, int n_samples, int j)
{
    for (int i = 0; i < n_samples; i++)
    {
        lookUpTable[j].min = lookUpTable[j].min < X[i][j] ? lookUpTable[j].min : X[i][j];
        lookUpTable[j].max = lookUpTable[j].max < X[i][j] ? lookUpTable[j].max : X[i][j];
    }
}

void ExtraTreesRandomSelector::GetRowMinMax(float16_t **X, int i_s, int i_e)
{
    for (int i = i_s; i < i_e && i < n_samples; ++i)
    {
        for (int j = 0; j < n_features; ++j)
        {
            while (!mut[j].try_lock())
                ;
            // SIMD?
            lookUpTable[j].min = lookUpTable[j].min < X[i][j] ? lookUpTable[j].min : X[i][j];
            lookUpTable[j].max = lookUpTable[j].max > X[i][j] ? lookUpTable[j].max : X[i][j];
            mut[j].unlock();
        }
    }
}

ExtraTreesRandomSelector::ExtraTreesRandomSelector(int n_features)
{
    generatedLimits = false;
    this->n_features = n_features;
    lookUpTable = new MinMax[n_features];
    mut = new mutex[n_features];
    n_threads = 4;
}

ExtraTreesRandomSelector::~ExtraTreesRandomSelector()
{
    delete[] lookUpTable;
    delete[] mut;
}

float16_t ExtraTreesRandomSelector::GetThreshold(float16_t **X, int *labels, int n_samples, int feature_index)
{
    auto start = std::chrono::high_resolution_clock::now();

    vector<thread> ths;
    int i_s, i_e;
    int samples_per_thread = n_samples / n_threads;
    i_s = 0;
    i_e = samples_per_thread;
    this->n_samples = n_samples;

    for (; i_e <= n_samples; i_s += samples_per_thread, i_e += samples_per_thread)
    {
        ths.emplace_back([this, X, i_s, i_e]()
                         { GetRowMinMax(X, i_s, i_e); });
    }

    for (auto &th : ths)
        th.join();

    float to_ret = ((rand() % 65504) / (float16_t)(65504)) * (lookUpTable[feature_index].max - lookUpTable[feature_index].min) + lookUpTable[feature_index].min;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    bool print = false;
    if (print)
        std::cout << "Time to get thresholds for ExtraTree: " << duration.count() << " ms" << std::endl;
    return to_ret;
}
