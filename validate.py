from Utils import *
from Model import image_embedder
from clustering import *


tfidf_dim=TEXT_VEC_SIZE
img_emb=2048
text_emb=0
out_classes=11014
config = {"n_epochs": 2,
          "validate_every_n_epoch": 1,
          "LR": 1e-4,
          "threshold": 0.3,
          "train_batch_size": 6,
          "valid_batch_size": 6,
          "device": device,
          }
#data_folder
log_file_name = "log_files/"+ __name__ + "@" +  now.strftime("%A_%H_%I_") + "log"
logging.basicConfig(level=logging.DEBUG,  format="%(asctime)s  %(levelname)s       %(message)s", handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()])
logging.info(f"TEXT_VEC_SIZE: {TEXT_VEC_SIZE}")
logging.info(f"img_emb: {img_emb}")
logging.info(f"text_emb: {text_emb}")

now = datetime.datetime.now()
run_name = "Valid@" + now.strftime("%A - %H:%I")
wandb.login()
set_all_seeds()

wandb_conf = config.copy()
wandb_conf.update({"IMG_SIZE" : IMAGE_SIZE, "img_emb": img_emb, "text_emb": text_emb})

for fold in range(2):
    with wandb.init(project="Shopee", config=wandb_conf, save_code=True, group="Valid", job_type="Fold" + str(fold + 1), name=run_name, mode="dryrun"):
        # ...Create train and validation for the fold
        train_ds, valid_ds = create_train_test(mode="validation", give_fold=fold)
        # ...initiate and train the model
        model = image_embedder(img_emb=img_emb, out_classes=train_ds.df.label_group.nunique())
        model.fit(train_ds, valid_ds, config)