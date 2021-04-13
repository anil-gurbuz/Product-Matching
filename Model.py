from lib import *
from ArcFace import ArcFace
from Base_Trainer import Base_model
from clustering import cosine_find_matches_cupy, euclidian_find_matches_cupy, get_best_threshold, matches_to_f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_trained_image_model_folder = "data/image_model/"

class image_embedder(Base_model):
    def __init__(self, tfidf_dim=TEXT_VEC_SIZE, img_emb=256, text_emb=256, out_classes=11014):
        super().__init__()

        try:
            self.effnet = EfficientNet.from_name("efficientnet-b1")
            self.effnet.load_state_dict(torch.load(pre_trained_image_model_folder + "efficientnet-b1-f1951068.pth"))
        except:
            self.effnet = EfficientNet.from_pretrained(
                pre_trained_image_model_folder + 'efficientnet-b1-f1951068.pth')  # Might try without last 2 layers -- basically with extracting the features

        self.linear1 = nn.Linear(1000, img_emb)
        self.linear2 = nn.Linear(tfidf_dim, text_emb)
        self.arcface_head = ArcFace(img_emb + text_emb, out_classes)

        self.loss = nn.CrossEntropyLoss()
        self.metric = metrics.accuracy_score


        self.threshold = None


    def forward(self, data_batch):
        images = data_batch["image"]
        text_vec = data_batch["text_vec"]
        label = data_batch["label"]

        images = self.effnet(images)
        img_emb = self.linear1(images)
        text_emb = self.linear2(text_vec)

        full_emb = torch.cat([img_emb,text_emb], dim=1)

        if self.training:
            out = self.arcface_head(full_emb, label)
            loss = self.loss(out, label)
            metric = 0

            return out, loss, metric
        else:
            return full_emb, 0, 0

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def validate_all(self, valid_dataset):
       embeddings = self.predict(valid_dataset, batch_size=self.valid_batch_size)

       # Find best threshold for cosine distance and log the f1
       best_cosine_threshold = get_best_threshold(cosine_find_matches_cupy, embeddings, valid_dataset.df.posting_id, valid_dataset.df.target, np.arange(0.05,0.5,0.05))
       matches = cosine_find_matches_cupy(embeddings, valid_dataset.df.posting_id, best_cosine_threshold,create_submission=False)
       f1_score = matches_to_f1_score(valid_dataset.df.target, pd.Series(matches))
       wandb.log({"Valid_cosine_F1": f1_score}, step=self.current_train_step)

