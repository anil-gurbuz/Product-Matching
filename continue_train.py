from Utils import *
from Model import image_embedder
from clustering import *
from Base_Trainer import load_snapshot

model_snapshot_path = "data/Snapshot_tfidf_15000_256_256_11014_10.pth"
mode = "debug" #  "train" or "debug"
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
# wandb_id =
#wandb.init(project="Shopee",resume = wandb_id)
wandb.init()

# # Go on training using a snapshot
train_ds, valid_ds = create_train_test(mode=mode)
model = image_embedder(tfidf_dim, img_emb, text_emb, out_classes)
model.to(device)
model = load_snapshot(model_snapshot_path,  model, device, optimizer_obj=torch.optim.Adam(model.parameters()), scheduler_obj=None)
model.fit(train_ds, valid_ds, {})