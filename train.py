from Utils import *
from Model import image_embedder
from clustering import *

mode = "full_train" # OR "baby_sit"
tfidf_dim=15000
img_emb=256
text_emb=256
out_classes=11014
config = {"n_epochs": 10,
          "validate_every_n_epoch": 5,
          "LR": 1e-4,
          "threshold": 0.3,
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "device": device,
          }


now = datetime.datetime.now()
run_name = mode + "@" + now.strftime("%A - %H:%I")
model_save_path = f"data/tfidf_{tfidf_dim}_{img_emb}_{text_emb}_{out_classes}.pth"

if mode == "baby_sit":
    config["validate_every_n_epoch"], config["n_epochs"] = 2, 2
wandb.login()
set_all_seeds()
with wandb.init(project="Shopee", config=config, save_code=True, group=mode, job_type="NA", name=run_name, mode="dryrun"):
    train_ds, valid_ds = create_train_test(mode=mode)
    model = image_embedder(tfidf_dim, img_emb, text_emb, out_classes)
    model.fit(train_ds, valid_ds, config)

    if mode=="full_train":
        torch.save(model.state_dict(), model_save_path)

