
#include <arm_neon.h>
#include "Tree.h"
#include <vector>
#include <set>
using namespace std;

class TreeTrainer
{
private:
    static TreeTrainer *instance;
    TreeTrainer() { instance = this; }
    vector<vector<pair<int, float16_t>>> splits;
    void getAllSplits(float16_t **X, int *labels, int n_trees, int n_select_features, int n_samples, int n_features, IFeatureSelector *selector);
    void fitTree(TreeNode *node, float16_t **X, int *labels, int depth, int max_depth, int n_samples, int n_features, int tree_index);
    void shake(int *features, int len);
    int getMajorityClass(int *labels, int n_samples);
    void splitData(float16_t **X, int *labels, int n_samples, int feature, float16_t threshold,
                   float16_t **&left_X, float16_t **&right_X, int *&left_labels, int *&right_labels,
                   int &left_size, int &right_size);
    int predictTree(TreeNode *root, float16_t *X);

public:
    static TreeTrainer *GetInstance()
    {
        if (instance == nullptr)
            instance = new TreeTrainer();
        return instance;
    }
    void fit(Tree **tree, float16_t **X, int *labels, int max_depth, int n_samples, int n_features, int n_trees, IFeatureSelector *selector, int n_threads);
    int *predict(Tree **tree, float16_t **X, int n_trees, int n_samples);
    void printTree(Tree *tree);
    void print(TreeNode *root);
    float getAccuracy(int *true_labels, int *predicted_labels, int n_samples);
};