#include "DataReader.h"
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

using namespace std;
DataReader::DataReader()
{
    n_samples = 0;
    n_features = 0;
    data = nullptr;
    generated = false;
    thread_count = 12;
    print_first = 0;
}

void DataReader::ReadData(const char *filepath, int n_samples, int n_features)
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
            data[i] = new float16_t[n_samples];
    }

    this->n_samples = n_samples;
    this->n_features = n_features;
    vector<thread> ths;
    int i_s, i_e;
    int entries_per_thread = n_samples / thread_count;
    i_s = 0;
    i_e = entries_per_thread;

    FILE *f = fopen(filepath, "r");
    if (f == nullptr)
    {
        cout << "File doesn't exist" << endl;
        exit(1);
    }

    // Create threads to generate data
    for (; i_e <= n_samples; i_s += entries_per_thread, i_e += entries_per_thread)
    {
        ths.emplace_back([this, i_s, i_e, filepath, n_features]()
                         { this->read(filepath, i_s, i_e, n_features); });
    }

    // Wait for threads
    for (auto &th : ths)
        th.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to read data: " << duration.count() << " ms" << std::endl;
}

void DataReader::read(const char *filepath, int i_start, int i_end, int n_features)
{
    // We have n_features float16 (16bits -> 2B) separated by n_features-1 characters(,) and with a '\n' char at the end -> n_features chars -> n_features B
    // Thus, each line has n_features*2 (floats) + n_features (characters) bytes => n_features*3 B
    // If we read from i_s to i_e => we read i_e-i_s rows
    FILE *f = fopen(filepath, "r");
    if (f == nullptr)
    {
        cout << "Thread couldn't find the file" << endl;
        exit(1);
    }

    int total_read_bytes = 3 * n_features * (i_end - i_start);
    char *read_bytes = new char[total_read_bytes];

    // We know that each row has 3 * n_features bytes, and the threads before have read i_start rows, so we offset read position by
    //  3 * n_features * i_start
    //  Seek inside the file
    fseek(f, (long)(3 * n_features * i_start), SEEK_SET);

    char *token;
    char *row;
    int j = 0;
    int i = i_start;

    fread(read_bytes, sizeof(char), total_read_bytes, f);

    row = strtok(read_bytes, "\n");
    while (row != NULL && i < i_end)
    {
        if (i == 0)
            goto label;
        j = 0;
        token = strtok(row, ",");
        while (token != NULL && j < n_features)
        {
            // Convert to float
            data[i][j] = strtof(token, NULL);
            token = strtok(nullptr, ",");
            ++j;
        }
    label:
        row = strtok(nullptr, "\n");
        ++i;
    }
    delete[] read_bytes;
    fclose(f);
}

void DataReader::deallocateMemory()
{
    cout << "Deleting" << endl;
    if (!generated)
        return;
    for (int i = 0; i < n_samples; ++i)
        delete[] data[i];
    delete[] data;
    delete[] labels;
}

DataReader::~DataReader()
{
    deallocateMemory();
}