import torch
import numpy as np
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

def link_prediction(node_embedding, edges_test, labels_test, edge_fun = "l2", dimensions = 13, num_partitions = 2, GCN = False):
    acc_test = []
    adjust = []
    edges_all = edges_test
    edge_labels_all = labels_test
    edge_fn = edge_functions[edge_fun]
    partitioner = model_selection.StratifiedKFold(num_partitions, shuffle=True)
    for train_inx, test_inx in partitioner.split(edges_all, edge_labels_all):
        edges_train = [edges_all[jj] for jj in train_inx]
        labels_train = [edge_labels_all[jj] for jj in train_inx]
        edges_test = [edges_all[jj] for jj in test_inx]
        labels_test = [edge_labels_all[jj] for jj in test_inx]

        edge_features_train = edges_to_features(node_embedding, edges_train, edge_fn, dimensions, GCN)
        edge_features_test = edges_to_features(node_embedding, edges_test, edge_fn, dimensions, GCN)

        # Linear classifier
        scaler = StandardScaler()
        lin_clf = LogisticRegression(C=1)
        clf = pipeline.make_pipeline(scaler, lin_clf)
        clf.fit(edge_features_train, labels_train)

        # NMI = metrics.scorer.normalized_mutual_info_scorer(clf, edge_features_test, labels_test))
        # auc_test = metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test))
        edge_features_test[np.isnan(edge_features_test)] = 0
        adjust.append(metrics._scorer.adjusted_mutual_info_scorer(clf, edge_features_test, labels_test))
        # recall_test = metrics.scorer.recall_scorer(clf, edge_features_test, labels_test))
        acc_test.append(metrics._scorer.accuracy_scorer(clf, edge_features_test, labels_test))

    return sum(acc_test)/len(acc_test), sum(adjust)/len(adjust)


def edges_to_features(nodes_embedding, edge_list, edge_function, dimensions, GCN=False):
    embedding = nodes_embedding
    n_tot = len(edge_list)
    feature_vec = np.empty((n_tot, dimensions), dtype='f')

    # Iterate over edges
    for ii in range(n_tot):
        v1, v2 = edge_list[ii]

        # Edge-node features
        if GCN:
            emb1 = embedding[int(v1)]
            emb2 = embedding[int(v2)]
            feature_vec[ii] = edge_function(emb1.detach().numpy(),emb2.detach().numpy())
        else:
            emb1 = embedding[str(v1)]
            emb2 = embedding[str(v2)]
            try:
                feature_vec[ii] = edge_function(emb1, emb2)
            except ValueError:
                print("")

    return feature_vec