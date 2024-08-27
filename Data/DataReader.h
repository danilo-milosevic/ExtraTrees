#include <arm_neon.h>
#include <iostream>
using namespace std;
class DataReader
{
private:
    int n_samples, n_features;
    int print_first; // debugging
    int max_label;
    bool generated;
    void read(const char *filepath, int i_start, int i_end, int n_features);
    void deallocateMemory();
    int thread_count;

public:
    float16_t **data;
    int *labels;
    DataReader();
    void ReadData(const char *filepath, int n_samples, int n_features);
    ~DataReader();
};