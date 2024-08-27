#include "Tree.h"
#include "ExtraTreesRandomSelector.h"
#include "TreeTrainer.h"
#include "IFeatureSelector.h"

class RandomForestClassifier
{
private:
    int n_trees;
    Tree **trees;
    IFeatureSelector *selector;
    TreeTrainer *trainer;

public:
    RandomForestClassifier(int n_trees)
    {
        this->n_trees = n_trees;
        trees = new Tree *[n_trees];
        trainer = TreeTrainer::GetInstance();
    }

    void fit(float16_t **X, int *labels, int n_samples, int n_features, int max_depth)
    {
        selector = new ExtraTreesRandomSelector(n_features);
        trainer->fit(trees, X, labels, max_depth, n_samples, n_features, n_trees, selector, 8);
    }
};