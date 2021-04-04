from lib import *

IMAGE_SIZE = (384, 384)
data_folder = "data"

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


def create_train_test(mode=None):
    train = pd.read_csv( data_folder + "/train.csv")
    train["file_path"] = train.image.map(lambda x: data_folder + "/train_images/" + str(x))
    # Create submission format targets
    train["target"] = create_submission_format(train)
    n_labels = train.label_group.nunique()

    if mode == "tiny_data":
        train_labels = train.label_group.unique()[np.random.randint(0, n_labels, 100)]
        valid_labels = train.label_group.unique()[np.random.randint(0, n_labels, 200)]
        valid_labels = [label for label in valid_labels if label not in train_labels]

        valid = train.loc[train.label_group.isin(valid_labels),].reset_index(drop=True)
        train = train.loc[train.label_group.isin(train_labels),].reset_index(drop=True)

        # Encode label_group for train only
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)

        return train, valid

    if mode == "validation":
        train_labels = train.label_group.unique()[np.random.randint(0, n_labels, n_labels // 3)]

        valid = train.loc[~train.label_group.isin(train_labels),].reset_index(drop=True)
        train = train.loc[train.label_group.isin(train_labels),].reset_index(drop=True)

        # Encode label_group for train only
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)

        return train, valid

    if mode == "full_train":
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)

        return train, None

    if mode == "inference":
        test = pd.read_csv(data_folder + "/test.csv")
        test["file_path"] = test.image.map(lambda x: data_folder + "/test_images/" + str(x))

        return test, None
