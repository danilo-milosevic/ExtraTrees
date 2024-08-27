#include <iostream>
#include "DataReader.h"
#include "DataGenerator.h"
#include <assert.h>

void testReadData()
{
    DataReader *reader = new DataReader();
    reader->ReadData("/Users/danilomilosevic/Documents/Danilo/ExtraTrees/Data/dataset.csv", 110000, 10);
    delete reader;
}

void testGenerateData()
{
    DataGenerator *gen = new DataGenerator();
    gen->GenerateData(100000, 10, 10, 5);
    delete gen;
}

int main()
{
    testReadData();
    testGenerateData();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
