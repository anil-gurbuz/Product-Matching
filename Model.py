from lib import *
from ArcFace import ArcFace
from Base_Trainer import Base_model
from clustering import cosine_find_matches_cupy, euclidian_find_matches_cupy, get_best_threshold, matches_to_f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_trained_image_model_folder = "data/image_model/"
model_file_name = "efficientnet-b0-355c32eb.pth"
model_name = "efficientnet-b0"

class image_embedder(Base_model):
    def __init__(self, img_emb=2000, out_classes=11014, dropout=0.3):
        super().__init__()

        try:
            self.effnet = EfficientNet.from_name(model_name)
            self.effnet.load_state_dict(torch.load(pre_trained_image_model_folder + model_file_name))
        except:
            self.effnet = EfficientNet.from_pretrained(pre_trained_image_model_folder + model_file_name)


        self.arcface_head = ArcFace(1000, out_classes)

        self.loss = nn.CrossEntropyLoss()
        self.metric = metrics.accuracy_score


        self.threshold = None


    def forward(self, data_batch):
        images = data_batch["image"]
        label = data_batch["label"]

        img_emb = self.effnet(images)

        if self.training:
            out = self.arcface_head(img_emb, label)
            loss = self.loss(out, label)
            metric = 0

            return out, loss, metric
        else:
            return img_emb, 0, 0

    def set_optimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

 #   def set_scheduler(self):
 #       self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)


    def validate_all(self, valid_dataset):
       embeddings = self.predict(valid_dataset, batch_size=self.valid_batch_size)

       # Find best threshold for cosine distance and log the f1
       best_cosine_threshold = get_best_threshold(cosine_find_matches_cupy, embeddings, valid_dataset.df.posting_id, valid_dataset.df.target, np.arange(0.05,0.30,0.05))
       matches = cosine_find_matches_cupy(embeddings, valid_dataset.df.posting_id, best_cosine_threshold,create_submission=False)
       f1_score = matches_to_f1_score(valid_dataset.df.target, pd.Series(matches))
       wandb.log({"Valid_cosine_F1": f1_score}, step=self.current_train_step)

