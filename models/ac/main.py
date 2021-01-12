from sklearn.cluster import AgglomerativeClustering
from models.utils.text_segmentation import get_start_indices


def find_style_change_starts_by_ac(feature_vectors, hyperparams):
    style_change_borders = [0]

    # used hyper parameters
    n_clusters = 3 if not hyperparams else hyperparams[0]
    affinity = 'euclidean' if not hyperparams else hyperparams[1]
    linkage = 'ward' if not hyperparams else hyperparams[2]

    if len(feature_vectors) < n_clusters:
        return []

    model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    clusters = model.fit_predict(feature_vectors)
    for i in range(len(clusters)-1):
        if clusters[i+1] != clusters[i]:
            style_change_borders.append(i+1)
    return style_change_borders if len(style_change_borders) != 1 else []


def make_prediction(feature_vectors, paragraphs, hyperparams):
    # finds borders by Agglomerative Clustering
    style_change_borders = find_style_change_starts_by_ac(feature_vectors,  hyperparams)

    # result format correction(found borders -> breaches)
    style_breaches = [] if not style_change_borders else get_start_indices(style_change_borders, paragraphs)
    result_dict = {"style_change": True if style_change_borders else False, "style_breaches": style_breaches}
    return result_dict
