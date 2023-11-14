from typing import List
from statistics import mode, mean

from PointSet import PointSet, FeaturesTypes

label_possible_values = [False, True]

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        leaves : Leaf
    """


    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.points = PointSet(features, labels, types)
        self.best_feature_index, _ = self.points.get_best_gain()
        self.best_feature = self.points.features[:, self.best_feature_index]
        self.best_feature_possible_values = set(self.best_feature)
        self.h = h

        self.leaves: Leaf = []
        
        if (h == 0 or set(labels) == 1 or len(labels) == 0 or self.best_feature_index == None): return

        for feature_value in self.best_feature_possible_values:
            mask_value = self.best_feature == feature_value
            features_i = self.points.features[mask_value]
            labels_i = self.points.labels[mask_value]
        
            if (len(features_i) != 0 and len(labels_i) != 0): 
                decision = mode(labels_i)

                self.leaves.append(
                    Leaf(
                        feature_value=feature_value,
                        decision=decision,
                        tree=Tree(
                            features=features_i,
                            labels=labels_i,
                            types=types,
                            h=h-1
                        )
                    )
                )

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        for leaf in self.leaves:
            if features[self.best_feature_index] == leaf.feature_value:
                if (len(leaf.tree.leaves) == 0): return leaf.decision
                else: return leaf.tree.decide(features)

class Leaf:
    def __init__(self,
                feature_value: float,
                decision: bool,
                tree: Tree):

        self.feature_value = feature_value
        self.decision = decision
        self.tree = tree
