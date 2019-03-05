import sklearn.metrics as metrics


def evaluate_(pred_labels, y):
    return (metrics.completeness_score(pred_labels, y),
            metrics.homogeneity_score(pred_labels, y),
            metrics.v_measure_score(pred_labels, y))


def evaluate(model, X, y):
    return evaluate_(model.fit_predict(X), y)
