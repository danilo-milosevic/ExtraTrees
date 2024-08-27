#include "ExtraTreesForest.h"
#include "../SelectFeature/ExtraTreesRandomSelector.h"
ExtraTreesForest::ExtraTreesForest(int treeCount)
{
    trees = new Tree *[treeCount];
    for (int i = 0; i < treeCount; ++i)
        trees[i] = new Tree();
    this->treeCount = treeCount;
    trainer = TreeTrainer::GetInstance();
}

void ExtraTreesForest::fit(float16_t **X, int *labels, int n_features, int n_samples, int max_depth)
{
    selector = new ExtraTreesRandomSelector(n_features);
    cout << "Start fitting..." << endl;
    trainer->fit(trees, X, labels, max_depth, n_samples, n_features, treeCount, selector, 1);
}

float ExtraTreesForest::predict_with_precision(float16_t **X, int *labels, int n_samples)
{
    int *predicted = trainer->predict(trees, X, treeCount, n_samples);
    float acc = trainer->getAccuracy(labels, predicted, n_samples);
    delete[] predicted;
    return acc;
}
