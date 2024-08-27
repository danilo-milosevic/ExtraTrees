#pragma once
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
    thread_count = 8;
}

void DataGenerator::generate(int i_start, int i_end, float16_t max_value)
{
    int j;
    for (int i = i_start; i < i_end && i < n_samples; ++i)
    {
        int r = (((double)rand()) / RAND_MAX) * max_label;
        labels[i] = r;
        j = 0;
        for (; j <= n_features - 8; j += 8)
        {
            uint16x8_t random_vals = {
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504),
                static_cast<uint16_t>(rand() % 65504)};

            float16x8_t float_vals = vcvtq_f16_u16(random_vals);
            float16x8_t normalized_vals = vmulq_n_f16(float_vals, mult);
            vst1q_f16(&data[i][j], normalized_vals);
        }
        // while (j < n_features)
        // {
        //     data[i][j] = (float16_t)(rand()) * mult;
        //     ++j;
        // }
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
    mult = max_value / (float16_t)65504;
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