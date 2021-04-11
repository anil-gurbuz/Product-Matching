from Utils import *
from Model import image_embedder
from clustering import *



now = datetime.datetime.now()
run_name = "Valid@" + now.strftime("%A - %H:%I")


config = {"n_epochs": 10,
          "validate_every_n_epoch": 5,
          "LR": 1e-4,
          "threshold": 0.3,
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "device": device,
          }

wandb.login()
set_all_seeds()


for fold in range(3):
    with wandb.init(project="Shopee", config=config, save_code=True, group="Valid", job_type="Fold" + str(fold + 1), name=run_name, mode="dryrun"):
        # ...Create train and validation for the fold
        train_ds, valid_ds = create_train_test(mode="validation", give_fold=fold)
        # ...initiate and train the model
        model = image_embedder(out_classes=train_ds.df.label_group.nunique())
        model.fit(train_ds, valid_ds, config)