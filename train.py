from Utils import *
from Model import image_embedder
from clustering import *
# JUST TO TRY MAKING CHANGE #

mode = "train" #  "train" or "debug"
tfidf_dim=TEXT_VEC_SIZE
img_emb=256
text_emb=256
out_classes=11014
config = {"n_epochs": 10,
          "validate_every_n_epoch": 1000000,
          "LR": 1e-4,
          "threshold": 0.25,
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "device": device,
          }


now = datetime.datetime.now()
run_name = mode + "@" + now.strftime("%A - %H:%I")
model_save_path = f"data/tfidf_{tfidf_dim}_{img_emb}_{text_emb}_{out_classes}_{config['n_epochs']}.pth"
model_snapshot_path = f"data/Snapshot_tfidf_{tfidf_dim}_{img_emb}_{text_emb}_{out_classes}_{config['n_epochs']}.pth"

if mode == "debug":
    config["validate_every_n_epoch"], config["n_epochs"] = 2, 2
wandb.login()
set_all_seeds()

wandb_conf = config.copy()
wandb_conf.update({"TEXT_VEC_SIZE" : TEXT_VEC_SIZE, "img_emb": img_emb, "text_emb": text_emb})

wandb_id = wandb.util.generate_id()
logging.info(f"Wandb id for resume training:{wandb_id}" )
with wandb.init(project="Shopee", config=wandb_conf, save_code=True, group=mode, job_type="NA", name=run_name, mode="dryrun", resume="allow", id = wandb_id):
    train_ds, valid_ds = create_train_test(mode=mode)
    model = image_embedder(tfidf_dim, img_emb, text_emb, out_classes)
    model.fit(train_ds, valid_ds, config)

    if mode=="train":
        torch.save(model.state_dict(), model_save_path)
        model.snapshot(model_snapshot_path, config)

