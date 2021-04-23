from Utils import *
from Model import image_embedder
from clustering import *
# JUST TO TRY MAKING CHANGE #

mode = "train" #  "train" or "debug"
img_emb=512
out_classes=11014
config = {"n_epochs": 100000,
          "validate_every_n_epoch": 2,
          "LR": 1e-4,
          "threshold": 0.25,
          "train_batch_size": 6,
          "valid_batch_size": 6,
          "device": device,
          "save_every_n_epoch" : 2
          }
if mode == "debug":
    config["validate_every_n_epoch"], config["n_epochs"] = 1,1

now = datetime.datetime.now()
run_name = mode + "@" + now.strftime("%A - %H:%I")
model_save_path = f"data/imageonly_{img_emb}_{out_classes}_{config['n_epochs']}.pth"
model_snapshot_path = f"data/snap_imageonly_{img_emb}_{out_classes}_{config['n_epochs']}.pth"

wandb.login()
set_all_seeds()
wandb_conf = config.copy()
wandb_conf.update({"img_emb": img_emb})
wandb_id = wandb.util.generate_id()
logging.info(f"Wandb id for resume training:{wandb_id}" )

with wandb.init(project="Shopee", config=wandb_conf, save_code=True, group=mode, job_type="NA", name=run_name, resume="allow", id = wandb_id): # mode="dryrun",
    train_ds, valid_ds = create_train_test(mode=mode)
    model = image_embedder(img_emb,  out_classes)

    model.fit(train_ds, valid_ds, config, model_save_path)

    if mode=="train":
        torch.save(model.state_dict(), model_save_path)
        model.snapshot(model_snapshot_path, config)

