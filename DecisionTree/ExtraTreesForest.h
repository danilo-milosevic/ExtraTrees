
#include "TreeTrainer.h"
#include "Tree.h"
#include "../SelectFeature/IFeatureSelector.h"
#include <arm_neon.h>
class ExtraTreesForest
{
private:
    TreeTrainer *trainer;
    int treeCount;
    IFeatureSelector *selector;

public:
    Tree **trees;
    ExtraTreesForest(int treeCount);
    void fit(float16_t **X, int *labels, int n_features, int n_samples, int max_depth);
    float predict_with_precision(float16_t **X, int *labels, int n_samples);
};