### ONLY USING IMAGE FOR NOW
# 0. Pre-process data before creating the dataset
        # Encode label_class & ####phash
# 1. Create Dataset class
# 2. Create Embeddings using transfer Learning
# 3. Cross-validate directly using validation sets generated features and see their submission error.
# In other words, no direct validation to class predictions instead clustering based validation


# 3. Create Inference
# 4. CV Loop

# THINGS TO try first
# 1. Monitor training to decide n_epochs
        # Loss at each epoch
        # Training f1 at every 50 epochs
        # Validation f1 at every 50 epochs
        ## NOT A MUST ### Would be great to have histogram of activations at each layer -- This is kinda tricky
# 2. Make a sample submission -- write an inference notebook
# 3. Add text information using bert and create embeddings together


from Utils import *
from Shopee_dataset import ShopeeDataset, get_transforms
from Model import image_embedder
from clustering import cluster


threshold_candidates = 10.0**np.arange(-0.8,0,0.1)
overfit = False
load = True
config = {"n_epochs" : 1000,
          "LR" : 1e-4,
          "threshold" : 0.2,
          "baby_sit" : True}

wandb.login()




set_all_seeds()
train, test = create_train_test()



if overfit:
    train = train.sample(n=50, random_state=0)



if config.baby_sit:
    n_epochs = 10
    train, valid = create_train_test(label_sample=100)
    print("Number of rows sampled ", train.shape[0])
    print("Number of rows in valid ", valid.shape[0])
    train_ds = ShopeeDataset(train, "train", transforms=get_transforms(), tokenizer=None)
    valid_ds = ShopeeDataset(valid, "test", transforms=get_transforms(), tokenizer=None)

    if not load:
        with wandb.init(project="Shopee", config=config):
            model = image_embedder(embed_size=128, out_classes=train.label_group.nunique())

            tracker = model.fit(train_ds, None, n_epochs=n_epochs, device=device, lr=config.LR)
            embeddings = model.predict(valid_ds, device)
            dist_matrix = cdist(embeddings, embeddings, "cosine")
            pickle.dump(dist_matrix, open("data/dist_matrix.pkl", "wb"))

    else:
        dist_matrix = pickle.load(open("data/dist_matrix.pkl", "rb"))

    best_threshold, threshold_scores = cluster(dist_matrix, threshold_candidates, valid_ds)

    print(threshold_scores)
    print("Best threshold:", best_threshold)
    print("best f1:", valid_ds.df.f1.mean())



else:
    trackers_dict = {}
    for fold in np.sort(train.fold.unique()):
        train_ds= ShopeeDataset(train.loc[train.fold!=fold,],  "train", transforms= get_transforms(), tokenizer=None)
        valid_ds = ShopeeDataset(train.loc[train.fold==fold,], "test", transforms=get_transforms(), tokenizer=None)

        model = image_embedder(embed_size=128, out_classes=train.label_group.nunique())
        trackers_dict["fold" + str(fold)] = model.fit(train_ds, None, n_epochs=config.n_epochs, device=device, lr=config.LR)
        embeddings = model.predict(valid_ds, device)
        best_threshold, threshold_scores = cluster(embeddings, threshold_candidates, valid_ds)

        print(threshold_scores)
        print("Best threshold:", best_threshold)
        print("best f1:", valid_ds.df.f1.mean())






# ... For each layer
# Distribution of weights
# Distribution of activations
# Distribution of gradients