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
load = False
config = {"n_epochs" : 1000,
          "LR" : 1e-4,
          "threshold" : 0.2,
          "baby_sit" : True,
          "f1_monitor_rate":50,
          "train_batch_size":16,
          "valid_batch_size":16,
          "device":device,
          "regular_validate": False}

wandb.login()

set_all_seeds()
train, test = create_train_test()

run_name = ""

if overfit:
    train = train.sample(n=50, random_state=0)



if config["baby_sit"]:
    config["n_epochs"] = 10
    config["f1_monitor_rate"] = 5

    train, valid = create_train_test(train_label_size=100, valid_label_size=300)
    print("Number of rows sampled ", train.shape[0])
    print("Number of rows in valid ", valid.shape[0])
    train_ds = ShopeeDataset(train, "train", transforms=get_transforms(), tokenizer=None)
    valid_ds = ShopeeDataset(valid, "test", transforms=get_transforms(), tokenizer=None)

    with wandb.init(project="Shopee", config=config, save_code=True, group="baby_sit", job_type="train"):
        model = image_embedder(embed_size=128, out_classes=train.label_group.nunique())

        model.fit(train_ds, valid_ds, config)



# ... For each layer
# Distribution of weights
# Distribution of activations
# Distribution of gradients