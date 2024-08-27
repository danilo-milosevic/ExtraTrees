struct TreeNode
{
    bool is_leaf;
    int feature_index;
    double threshold;
    int label;
    TreeNode *left;
    TreeNode *right;
    TreeNode()
        : is_leaf(false), feature_index(-1), threshold(0.0), label(-1), left(nullptr), right(nullptr) {};
};