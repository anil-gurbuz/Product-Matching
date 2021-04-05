# 1. Babysit for deciding embed_size
# 2. Babysit for deciding LR
# Extra things to monitor:
# a. Weight Update magnitude /  Weight ratio
# b. Activation distribution per layer
# c. First Layer Visualisation for image


from Utils import *
from Model import image_embedder
from clustering import embeddings_to_submission

mode = "full_train"  # "tiny_data", "validation", "full_train", "inference"
save = True
baby_sit = False

os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_MODE"] = "dryrun" # "dryrun", "online" #wandb sync DIRECTORY to upload to server

now = datetime.datetime.now()
run_name = "model@" + now.strftime("%A - %H:%I")
job_type = "NA"
#wandb_mode = "online" # "online", "offline", "disabled" "dryrun"

config = {"n_epochs": 5,
          "LR": 1e-4,
          "threshold": 0.3,
          "f1_monitor_rate": 50,
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "device": device,
          "regular_validate": False,
          "embed_size": 256,
          "baby_sit":baby_sit
          }
wandb.login()
set_all_seeds()

if baby_sit:
    config["n_epochs"] = 3
    config["f1_monitor_rate"] = 3

if mode == "validation":
    # For each fold...
    for fold in range(3):
        with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type="Fold" + str(fold + 1),
                        name=run_name):
            # ...Create train and validation for the fold
            train_ds, valid_ds = create_train_test(mode=mode, fold=fold)
            # ...initiate and train the model
            model = image_embedder(embed_size=config["embed_size"], out_classes=train_ds.df.label_group.nunique())
            model.fit(train_ds, valid_ds, config)
            # ...Save the model
            if save:
                torch.save(model.state_dict(), data_folder + "/Embedder_" + mode + "Fold" + str(fold + 1) + ".pth")

if mode == "tiny_data" or mode == "full_train":
    with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type=job_type, name=run_name):
        train_ds, valid_ds = create_train_test(mode=mode)
        model = image_embedder(embed_size=config["embed_size"], out_classes=train_ds.df.label_group.nunique())
        model.fit(train_ds, valid_ds, config)
        if save:
            torch.save(model.state_dict(), data_folder + "/Embedder_" + mode + str(config["embed_size"])  + ".pth")

if mode == "inference":
    with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type=job_type,
                    name=run_name):
        test_ds, _ = create_train_test(mode=mode)
        model = image_embedder(embed_size=config["embed_size"], out_classes=train_ds.df.label_group.nunique())
        model.load_state_dict(torch.load(data_folder + "/Embedder_" + mode + str(config["embed_size"])  + ".pth"))
        embeddings = model.predict(test_ds, device)
        embeddings_to_submission(test_ds, embeddings, config["threshold"])



# ... For each layer
# Distribution of weights
# Distribution of activations
# Distribution of gradients



