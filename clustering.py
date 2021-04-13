# Can create more distance metrics and try them as well
from lib import *
from Utils import matches_to_f1_score


def get_best_threshold(method, embeddings, posting_ids, correct_matches, candidates):

    scores = dict()
    for threshold in candidates:

        matches = method(embeddings, posting_ids, threshold, create_submission=False)

        scores[threshold] = matches_to_f1_score(pd.Series(matches), pd.Series(correct_matches))

        logging.info(f"Method:{method.__name__},   Threshold:{threshold},   F1-Score: {scores[threshold]}")

    best_threshold = max(scores, key=scores.get)
    logging.info(f"Best Threshold:{best_threshold},  Best F1-Score: {scores[best_threshold]}")

    return best_threshold


def cosine_find_matches_cupy(embeddings, posting_ids, threshold, create_submission=True):
    embeddings = cp.array(embeddings)
    embeddings =  embeddings / cp.linalg.norm(embeddings, axis=1)[:,None]
    N = embeddings.shape[0]
    matches = []

    for i in tqdm(range(N)):
        v = embeddings[i, :]
        thresholded_bool = 1 - cp.dot(embeddings,v) < threshold
        thresholded_ix = cp.argwhere(thresholded_bool).squeeze(-1)
        thresholded_ix = thresholded_ix.get()
        match = " ".join(posting_ids[thresholded_ix])
        matches.append(match)

    if create_submission:
        sub = pd.DataFrame({"posting_id": posting_ids, "matches": matches})
        sub.to_csv("submission.csv", index=False)

    return matches


def cosine_find_matches_scipy_iterative(embeddings, posting_ids, threshold, create_submission=True):

    matches = []
    for idx in tqdm(range(embeddings.shape[0])):
        selected = cdist(embeddings[idx, np.newaxis], embeddings, "cosine")[0] < threshold
        matches.append(" ".join(posting_ids[selected].tolist()))

    if create_submission:
        sub = pd.DataFrame({"posting_id": posting_ids, "matches": matches})
        sub.to_csv("submission.csv", index=False)

    return matches

def cosine_find_matches_scipy_at_once(embeddings, posting_ids, threshold, create_submission=True):
    dist_matrix = cdist(embeddings, embeddings, "cosine")

    selected = dist_matrix < threshold
    matches = []
    for row in selected:
        matches.append(" ".join(posting_ids[row].tolist()))

    if create_submission:
        sub = pd.DataFrame({"posting_id": posting_ids, "matches": matches})
        sub.to_csv("submission.csv", index=False)

    return matches




# from cuml import metrics
def cosine_find_matches_cuml(embeddings, posting_ids, threshold, create_submission=True, mul=1000):
    matches_dict = {}
    n = embeddings.shape[0]
    steps = n // mul # mul is

    if(n % mul != 0):
        steps += 1

    for i in tqdm(range(steps)):
        a = i*mul
        b = (i+1)*mul
        b = min(b, n)
        k = metrics.pairwise_distances(
            embeddings[a:b], embeddings, metric='cosine')
        k = 1-k
        rows, cols = np.where(k > threshold)

        for (i, j) in zip(rows, cols):
            i = i+a
            tmp = posting_ids[j]
            if(i not in matches_dict.keys()):
                t = tmp
            else:
                t = matches_dict[i] + ' ' + tmp
            matches_dict[i] = t

    sub = pd.DataFrame({"posting_id": list(matches_dict.keys()), "matches": list(matches_dict.values())})
    sub = sub.set_index("posting_id").reindex(index=posting_ids).reset_index(drop=True)

    if create_submission:
        sub.to_csv("submission.csv", index=False)

    return sub.matches.to_list()



def euclidian_find_matches_cupy(embeddings, posting_ids, threshold, create_submission=True):
    embeddings = cp.array(embeddings)
    N = embeddings.shape[0]
    matches = []

    for i in tqdm(range(N)):
        v = embeddings[i, :]
        thresholded_bool = cp.linalg.norm(embeddings-v, axis=1) < threshold
        thresholded_ix = cp.argwhere(thresholded_bool).squeeze(-1)
        thresholded_ix = thresholded_ix.get()
        match = " ".join(posting_ids[thresholded_ix])
        matches.append(match)

    if create_submission:
        sub = pd.DataFrame({"posting_id": posting_ids, "matches": matches})
        sub.to_csv("submission.csv", index=False)

    return matches
