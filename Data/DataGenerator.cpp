#include "DataGenerator.h"
#include <iostream>
#include <chrono>
#include <arm_neon.h>
#include <vector>
#include <thread>

using namespace std;
DataGenerator::DataGenerator()
{
    n_samples = 0;
    n_features = 0;
    data = nullptr;
    generated = false;
    thread_count = 10;
}

void DataGenerator::generate(int i_start, int i_end, float16_t max_value)
{
    int j = 0;
    for (int i = i_start; i < i_end; ++i)
    {
        for (; j <= n_features - 8; j += 8)
        {
            labels[i] = rand() % max_label;
            uint16x8_t random_vals = {
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand()),
                static_cast<uint16_t>(rand())};
            float16x8_t float_vals = vcvtq_f16_u16(random_vals);
            float16x8_t normalized_vals = vmulq_n_f16(float_vals, (float16_t)(max_value / RAND_MAX));
            vst1q_f16(&data[i][j], normalized_vals);
        }
    }
}

void DataGenerator::deallocateMemory()
{
    cout << "Deleting" << endl;
    if (!generated)
        return;
    for (int i = 0; i < n_samples; ++i)
        delete[] data[i];
    delete[] data;
    delete[] labels;
}

void DataGenerator::GenerateData(int n_samples, int n_features, float16_t max_value, int max_label)
{
    auto start = std::chrono::high_resolution_clock::now();

    // If dimensions are wrong but the memory has already been allocated, deallocate it
    if (generated && (n_samples != this->n_samples || n_features != this->n_features))
    {
        deallocateMemory();
        generated = false;
    }

    // Allocate memory if it isn't already
    if (!generated)
    {
        float total_size_data = (n_features * sizeof(float16_t) * n_samples) / 1024.0;
        float total_size_labels = (sizeof(int) * n_samples) / 1024.0;
        cout << "Reserved " << total_size_data + total_size_labels << " MB" << endl;
        generated = true;
        data = new float16_t *[n_samples];
        labels = new int[n_samples];
        for (int i = 0; i < n_samples; ++i)
        {
            data[i] = new float16_t[n_samples];
        }
    }

    this->n_samples = n_samples;
    this->n_features = n_features;
    this->max_label = max_label;
    vector<thread> ths;
    int i_s, i_e;
    int samples_per_thread = n_samples / thread_count;
    i_s = 0;
    i_e = samples_per_thread;

    // Create threads to generate data
    for (; i_e <= n_samples; i_s += samples_per_thread, i_e += samples_per_thread)
    {
        ths.emplace_back([this, i_s, i_e, max_value]()
                         { generate(i_s, i_e, max_value); });
    }

    // Wait for threads
    for (auto &th : ths)
        th.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to generate data: " << duration.count() << " ms" << std::endl;
}
DataGenerator::~DataGenerator()
{
    deallocateMemory();
}