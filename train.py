# 1. Babysit for deciding embed_size
# 2. Babysit for deciding LR
# Extra things to monitor:
# a. Weight Update magnitude /  Weight ratio
# b. Activation distribution per layer
# c. First Layer Visualisation for image


from Utils import *
from Shopee_dataset import ShopeeDataset, get_transforms
from Model import image_embedder
from clustering import cluster

mode = "validation"  # "tiny_data", "validation", "full_train", "inference"

now = datetime.datetime.now()
run_name = "model@" + now.strftime("%A - %H:%I")
job_type = "NA"

config = {"n_epochs": 1000,
          "LR": 1e-4,
          "threshold": 0.2,
          "baby_sit": True,
          "f1_monitor_rate": 50,
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "device": device,
          "regular_validate": False,
          "embed_size": 256}
wandb.login()
set_all_seeds()

if config["baby_sit"]:
    config["n_epochs"] = 3
    config["f1_monitor_rate"] = 3

if mode == "validation":
    # For each fold...
    for fold in range(3):
        with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type="Fold" + str(fold + 1),
                        name=run_name):
            # ...Create train and validation for the fold
            train_ds, valid_ds = create_train_test(mode=mode, give_fold=fold)
            # ...initiate and train the model
            model = image_embedder(embed_size=config["embed_size"], out_classes=train_ds.df.label_group.nunique())
            model.fit(train_ds, valid_ds, config)
            # ...Save the model
            torch.save(model.state_dict(), data_folder + "/Embedder_" + mode + "Fold" + str(fold + 1) + ".pth")

if mode == "tiny_data" or mode == "full_train":
    with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type=job_type, name=run_name):
        train_ds, valid_ds = create_train_test(mode=mode, give_fold=fold)
        model = image_embedder(embed_size=config["embed_size"], out_classes=train_ds.df.label_group.nunique())
        model.fit(train_ds, valid_ds, config)
        torch.save(model.state_dict(), data_folder + "/Embedder_" + mode + ".pth")

if mode == "inference":
    with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type=job_type,
                    name=run_name):
        train_ds, valid_ds = create_train_test(mode=mode, give_fold=fold)
    pass

# ... For each layer
# Distribution of weights
# Distribution of activations
# Distribution of gradients
