#include <iostream>
#include "Data/DataGenerator.h"
using namespace std;
int main()
{
    DataGenerator *dataGenerator = new DataGenerator();
    dataGenerator->GenerateData(100000, 20, 5, 3);
    cout << "hej" << endl;
    delete dataGenerator;
    return 0;
}
