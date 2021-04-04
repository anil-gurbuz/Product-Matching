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


mode = "validation"     #"tiny_data", "validation", "full_train", "inference"
wandb_job_type = "Embed_Size_Search"

config = {"n_epochs": 1000,
          "LR": 1e-4,
          "threshold": 0.2,
          "baby_sit": True,
          "f1_monitor_rate": 50,
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "device": device,
          "regular_validate": False,
          "embed_size":256}
wandb.login()
set_all_seeds()



if config["baby_sit"]:
    config["n_epochs"] = 10
    config["f1_monitor_rate"] = 5

# Create datasets
train, valid = create_train_test(mode=mode)
print("Number of rows sampled ", train.shape[0])
print("Number of rows in valid ", valid.shape[0])
train_ds = ShopeeDataset(train, "train", transforms=get_transforms(), tokenizer=None)
if valid is not None:
    valid_ds = ShopeeDataset(valid, "test", transforms=get_transforms(), tokenizer=None)

# Fit and save model
with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type=wandb_job_type):
    model = image_embedder(embed_size=config["embed_size"], out_classes=train.label_group.nunique())
    model.fit(train_ds, valid_ds, config)
    torch.save(model.state_dict(), data_folder + "/Embedder_" + mode + ".pth")

# ... For each layer
# Distribution of weights
# Distribution of activations
# Distribution of gradients
