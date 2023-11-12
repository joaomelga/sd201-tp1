from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        true_labels_count = 0

        for label in self.labels:
            true_labels_count += 1 if label == True else 0
            
        num_of_labels = len(self.labels)
        
        if not num_of_labels: return 1
        
        prob_of_true = true_labels_count / num_of_labels
        prob_of_false = (num_of_labels - true_labels_count) / num_of_labels
        
        gini = 1 - (prob_of_true**2 + prob_of_false**2)
        
        return gini

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        num_of_features = len(self.features[0])
        features_w_labels = np.column_stack((self.labels, self.features))
        gini_gains = np.zeros(num_of_features, dtype=float)
        
        for feature in range(num_of_features):
            mask_true = features_w_labels[:, feature + 1] == True
            mask_false = features_w_labels[:, feature + 1] == False

            point_set_1 = PointSet(
                features = features_w_labels[mask_true][:, -num_of_features:],
                labels = features_w_labels[mask_true][:, 0],
                types = self.types
            )
            
            point_set_2 = PointSet(
                features = features_w_labels[mask_false][:, -num_of_features:],
                labels = features_w_labels[mask_false][:, 0],
                types = self.types
            )

            gini_split = (len(point_set_1.labels) * point_set_1.get_gini() + len(point_set_2.labels) * point_set_2.get_gini()) / len(self.labels)
            gini_gain = self.get_gini() - gini_split
            
            gini_gains[feature] = gini_gain
        
        best_gini_gain_index = np.argmax(gini_gains)

        return (best_gini_gain_index, gini_gains[best_gini_gain_index])
            
            
