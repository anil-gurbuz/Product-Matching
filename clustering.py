# Can create more distance metrics and try them as well
from lib import *
from Utils import matches_to_f1_score

def cluster(dist_matrix, threshold_candidates, ds):
    best_threshold, threshold_scores = _search_dist_threshold(dist_matrix, threshold_candidates, ds.df)

    # Create preds from distance matrix
    selected = dist_matrix < best_threshold
    matches = []
    for row in selected:
        matches.append(" ".join(ds.df.posting_id[row].tolist()))

    ds.df["preds"] = matches
    ds.df["f1"] = matches_to_f1_score(pd.Series(matches), pd.Series(ds.df.target.values))

    return best_threshold, threshold_scores

def _search_dist_threshold(dist_matrix, candidates, df ):

    scores = dict()
    for threshold in candidates:
        selected = dist_matrix < threshold

        matches = []
        for row in selected:

            matches.append(" ".join(df.posting_id[row].tolist()))

        scores[threshold] = matches_to_f1_score(pd.Series(matches), pd.Series(df.target.values))


    best_threshold = max(scores, key=scores.get)

    return best_threshold, scores


def embeddings_to_submission(ds, embeddings, threshold):
    dist_matrix = cdist(embeddings, embeddings, "cosine")

    selected = dist_matrix < threshold
    matches = []
    for row in selected:
        matches.append(" ".join(ds.df.posting_id[row].tolist()))

    ds.df["matches"] = matches

    ds.df[["posting_id", "matches"]].to_csv("submission.csv", index=False)


def iterative_embedding_to_submission(ds, embeddings, threshold):
    matches = []
    for idx in tqdm(range(embeddings.shape[0])):
        selected = cdist(embeddings[idx, np.newaxis], embeddings, "cosine")[0] < threshold
        matches.append(" ".join(ds.df.posting_id[selected].tolist()))

    ds.df["matches"] = matches
    ds.df[["posting_id", "matches"]].to_csv("submission.csv", index=False)