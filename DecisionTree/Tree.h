#pragma once
#include "TreeNode.h"
#include "../SelectFeature/IFeatureSelector.h"
struct Tree
{
    TreeNode *root;
    IFeatureSelector *featureSelector;
    Tree() : root(new TreeNode()), featureSelector(nullptr) {};
};