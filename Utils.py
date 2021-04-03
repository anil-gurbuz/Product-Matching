from lib import *

IMAGE_SIZE = (384, 384)


def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set True to be faster -- Check what is this


def create_submission_format(df):
    tmp = df.groupby("label_group").posting_id.unique().to_dict()
    matches = df.label_group.map(lambda x: " ".join(tmp[x]))
    return matches


def matches_to_f1_score(y_true, y_pred, mean=True):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))

    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = y_pred.apply(lambda x: len(x)).values - tp
    fn = y_true.apply(lambda x: len(x)).values - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    if mean:
        f1 = f1.mean()

    return f1


def create_train_test(train_label_size=None, valid_label_size=None):
    train = pd.read_csv("data/train.csv")
    train["file_path"] = train.image.map(lambda x: "data/train_images/" + str(x))
    # Create submission format targets
    train["target"] = create_submission_format(train)

    if train_label_size:
        n_labels = train.label_group.nunique()
        train_labels = train.label_group.unique()[np.random.randint(0, n_labels, train_label_size)]
        valid_labels = train.label_group.unique()[np.random.randint(0, n_labels, valid_label_size)]
        valid_labels = [label for label in valid_labels if label not in train_labels]

        valid = train.loc[train.label_group.isin(valid_labels),].reset_index(drop=True)
        train = train.loc[train.label_group.isin(train_labels),].reset_index(drop=True)

        # Encode label_group for train only
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)

        return train, valid


    cv_splitter = GroupKFold(n_splits=5)
    train["fold"]=-1

    # Assign folds for validation
    for fold, (train_idx, valid_idx) in enumerate(cv_splitter.split(train, None, train.label_group)):
        train.loc[valid_idx,"fold"] = fold


    test = pd.read_csv("data/test.csv")
    test["file_path"] = test.image.map(lambda x: "data/test_images/" + str(x))


    # Encode label_group
    enc = LabelEncoder()
    train.label_group = enc.fit_transform(train.label_group)

    return train, test
