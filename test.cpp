#pragma once
#include <iostream>
#include "Data/DataGenerator.h"
#include "DecisionTree/ExtraTreesForest.h"
using namespace std;
int main()
{
    DataGenerator *dataGenerator = new DataGenerator();
    dataGenerator->GenerateData(100000, 16, 5, 2);
    ExtraTreesForest *etf = new ExtraTreesForest(300);
    etf->fit(dataGenerator->data, dataGenerator->labels, 20, 100000, 10);
    float rez = etf->predict_with_precision(dataGenerator->data, dataGenerator->labels, 100000);
    cout << "Accuracy: " << rez << endl;
    delete etf;
    delete dataGenerator;
    return 0;
}
