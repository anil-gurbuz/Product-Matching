from lib import *
from Shopee_dataset import ShopeeDataset, get_transforms


def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set True to be faster -- Check what is this

# Function to get our text title embeddings
def get_text_embeddings(df,  max_features = TEXT_VEC_SIZE):

    vectorizer = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = vectorizer.fit_transform(df.title)
    del vectorizer
    return text_embeddings

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


def create_train_test(mode=None, give_fold=None):
    train = pd.read_csv( data_folder + "/train.csv")
    train["file_path"] = train.image.map(lambda x: data_folder + "/train_images/" + str(x))
    # Create submission format targets
    train["target"] = create_submission_format(train)
    n_labels = train.label_group.nunique()

    if mode == "debug":
        train_labels = train.label_group.unique()[np.random.randint(0, n_labels, 100)]
        valid_labels = train.label_group.unique()[np.random.randint(0, n_labels, 200)]
        valid_labels = [label for label in valid_labels if label not in train_labels]

        valid = train.loc[train.label_group.isin(valid_labels),].reset_index(drop=True)
        train = train.loc[train.label_group.isin(train_labels),].reset_index(drop=True)

        # Encode label_group for train only
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)

        train_ds = ShopeeDataset(train, "train", transforms=get_transforms())
        valid_ds = ShopeeDataset(valid, "test", transforms=get_transforms())

        return train_ds, valid_ds

    if mode == "validation":
        cv_splitter = GroupKFold(n_splits=2)
        train["fold"] = -1

        # Assign folds for validation
        for fold, (train_idx, valid_idx) in enumerate(cv_splitter.split(train, None, train.label_group)):
            train.loc[valid_idx, "fold"] = fold


        fold_train = train.loc[train.fold==give_fold,]
        fold_valid = train.loc[train.fold!=give_fold,]

        # Encode only current fold labels
        enc = LabelEncoder()
        fold_train.label_group = enc.fit_transform(fold_train.label_group)

        train_ds = ShopeeDataset(fold_train, "train", transforms=get_transforms())
        valid_ds = ShopeeDataset(fold_valid, "test", transforms=get_transforms())


        return train_ds, valid_ds


    if mode == "train":
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)
        train_ds = ShopeeDataset(train, "train", transforms=get_transforms())


        cv_splitter = GroupKFold(n_splits=7)
        train["fold"] = -1
        # Assign folds for validation
        for fold, (train_idx, valid_idx) in enumerate(cv_splitter.split(train, None, train.label_group)):
            train.loc[valid_idx, "fold"] = fold

        fold_valid = train.loc[train.fold == 1,]
        valid_ds = ShopeeDataset(fold_valid, "test", transforms=get_transforms())

        return train_ds, valid_ds

    if mode == "inference":
        test = pd.read_csv(data_folder + "/test.csv")
        test["file_path"] = test.image.map(lambda x: data_folder + "/test_images/" + str(x))

        test_ds = ShopeeDataset(test, "test", transforms=get_transforms())

        return test_ds, None


    if mode=="overfit":
        enc = LabelEncoder()
        train.label_group = enc.fit_transform(train.label_group)
        train = train.loc[train.label_group.isin(train.label_group.unique()[0:5]),]
        train_ds = ShopeeDataset(train, "train", transforms=get_transforms())
        return train_ds, None

