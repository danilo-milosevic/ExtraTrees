
#include "TreeTrainer.h"
#include <iterator>
#include <thread>
#include <iostream>

TreeTrainer *TreeTrainer::instance = nullptr;

void TreeTrainer::shake(int *features, int len)
{
    int z;
    for (int i = 0; i < len; ++i)
    {
        int index = rand() % len;
        z = features[i];
        features[i] = features[index];
        features[index] = z;
    }
}

int TreeTrainer::getMajorityClass(int *labels, int n_samples)
{
    // Create a map to count the occurrences of each label
    std::unordered_map<int, int> labelCount;
    for (int i = 0; i < n_samples; ++i)
    {
        labelCount[labels[i]]++;
    }
    // Find the label with the maximum count
    int majorityLabel = -1;
    int maxCount = 0;
    for (const auto &pair : labelCount)
    {
        if (pair.second > maxCount)
        {
            maxCount = pair.second;
            majorityLabel = pair.first;
        }
    }

    return majorityLabel;
}

void TreeTrainer::getAllSplits(float16_t **X, int *labels, int n_trees, int n_select_features, int n_samples, int n_features, IFeatureSelector *selector)
{
    // First initialize the feature indices (names)
    splits.clear();
    int features[n_features];
    for (int i = 0; i < n_features; ++i)
        features[i] = i;

    // For every tree we train
    for (int i = 0; i < n_trees; ++i)
    {
        // Find the splits
        splits.push_back(vector<pair<int, float16_t>>());
        // We take random order of features
        shake(features, n_features);
        // We don't always take all features
        for (int j = 0; j < n_select_features; ++j)
            splits.at(i).push_back(pair<int, float16_t>(features[j], selector->GetThreshold(X, labels, n_samples, features[j])));
    }
}

void TreeTrainer::splitData(float16_t **X, int *labels, int n_samples, int feature, float16_t threshold,
                            float16_t **&left_X, float16_t **&right_X, int *&left_labels, int *&right_labels,
                            int &left_size, int &right_size)
{
    std::vector<int> leftIndices, rightIndices;

    for (int i = 0; i < n_samples; ++i)
    {
        if (X[i][feature] <= threshold)
        {
            leftIndices.push_back(i);
        }
        else
        {
            rightIndices.push_back(i);
        }
    }
    left_size = leftIndices.size();
    right_size = rightIndices.size();

    left_X = new float16_t *[left_size];
    right_X = new float16_t *[right_size];
    left_labels = new int[left_size];
    right_labels = new int[right_size];

    for (int i = 0; i < left_size; ++i)
    {
        left_X[i] = X[leftIndices[i]];
        left_labels[i] = labels[leftIndices[i]];
    }

    for (int i = 0; i < right_size; ++i)
    {
        right_X[i] = X[rightIndices[i]];
        right_labels[i] = labels[rightIndices[i]];
    }
}

int TreeTrainer::predictTree(TreeNode *root, float16_t *X)
{
    if (root->is_leaf)
    {
        return root->label;
    }
    int target_feature = root->feature_index;
    if (X[target_feature] <= root->threshold)
    {
        predictTree(root->left, X);
    }
    else
        predictTree(root->right, X);
}

void TreeTrainer::fitTree(TreeNode *node, float16_t **X, int *labels, int depth, int max_depth, int n_samples, int n_features, int tree_index)
{
    if (depth >= max_depth)
    {
        node->is_leaf = true;
        node->label = getMajorityClass(labels, n_samples);
        return;
    }

    int feature_index = splits[tree_index][depth].first;
    float16_t threshold = splits[tree_index][depth].second;
    int left_size = 0, right_size = 0;
    float16_t **left_X = nullptr, **right_X = nullptr;
    int *left_labels = nullptr, *right_labels = nullptr;

    splitData(X, labels, n_samples, feature_index, threshold, left_X, right_X, left_labels, right_labels, left_size, right_size);

    if (left_size == 0 || right_size == 0)
    {
        node->is_leaf = true;
        node->label = getMajorityClass(labels, n_samples);
        return;
    }

    node->feature_index = feature_index;
    node->threshold = threshold;

    node->left = new TreeNode();
    node->right = new TreeNode();

    fitTree(node->left, left_X, left_labels, depth + 1, max_depth, left_size, n_features, tree_index);
    fitTree(node->right, right_X, right_labels, depth + 1, max_depth, right_size, n_features, tree_index);

    delete[] left_X;
    delete[] right_X;
    delete[] left_labels;
    delete[] right_labels;
}

void TreeTrainer::fit(Tree **tree, float16_t **X, int *labels, int max_depth, int n_samples, int n_features, int n_trees, IFeatureSelector *selector, int n_threads)
{
    auto start = std::chrono::high_resolution_clock::now();
    max_depth = min(max_depth, n_features);
    getAllSplits(X, labels, n_trees, max_depth, n_samples, n_features, selector);

    vector<thread> ths;
    int i_s, i_e;
    int trees_per_thread = n_trees / n_threads;
    i_s = 0;
    i_e = trees_per_thread;

    for (; i_e <= n_samples; i_s += trees_per_thread, i_e += trees_per_thread)
    {
        ths.emplace_back([this, X, tree, i_s, i_e, labels, max_depth, n_samples, n_features, n_trees]()
                         {
                            for (int i=i_s;i<i_e && i<n_trees;++i)
                                this->fitTree(tree[i]->root, X, labels, 0, max_depth, n_samples, n_features, i); });
    }

    for (auto &th : ths)
        th.join();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Time to fit " << n_trees << " trees: " << duration.count() << " s" << std::endl;
}

int *TreeTrainer::predict(Tree **tree, float16_t **X, int n_trees, int n_samples)
{
    int **predictions = new int *[n_samples];
    int *final_preds = new int[n_samples];
    for (int i = 0; i < n_samples; ++i)
    {
        predictions[i] = new int[n_trees];
    }

    for (int i = 0; i < n_samples; ++i)
        for (int j = 0; j < n_trees; ++j)
            predictions[i][j] = predictTree(tree[j]->root, X[i]);

    for (int i = 0; i < n_samples; ++i)
        final_preds[i] = getMajorityClass(predictions[i], n_trees);

    for (int i = 0; i < n_samples; ++i)
        delete[] predictions[i];
    delete[] predictions;

    return final_preds;
}

void TreeTrainer::print(TreeNode *root)
{
    if (root == nullptr)
        return;

    if (!root->is_leaf)
        cout << "Feature" << root->feature_index << " threshold: " << root->threshold << " | ";
    else
        cout << "Leaf!" << root->label << " class | ";
    print(root->left);
    print(root->right);
}

float TreeTrainer::getAccuracy(int *true_labels, int *predicted_labels, int n_samples)
{
    int correct = 0;
    for (int i = 0; i < n_samples; ++i)
        if (true_labels[i] == predicted_labels[i])
            correct++;
    return (float)correct / n_samples;
}

void TreeTrainer::printTree(Tree *tree)
{
    print(tree->root);
    cout << endl;
}
