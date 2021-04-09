from lib import *
from ArcFace import ArcFace
from Base_Trainer import Base_model
from Base_Trainer import Tracker
from clustering import cluster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class image_embedder(Base_model):
    def __init__(self, tfidf_dim, embed_size=1280, out_classes=11014):
        super().__init__()

        try:
            self.effnet = EfficientNet.from_name("efficientnet-b1")
            self.effnet.load_state_dict(torch.load("data/image_model/efficientnet-b1-f1951068.pth"))
        except:
            self.effnet = EfficientNet.from_pretrained(
                'data/image_model/efficientnet-b1-f1951068.pth')  # Might try without last 2 layers -- basically with extracting the features

        self.linear1 = nn.Linear(1000, embed_size)
        self.linear2 = nn.Linear(tfidf_dim, embed_size)
        self.arcface_head = ArcFace(embed_size*2, out_classes)

        self.f1_monitor_rate = None
        self.threshold = None
        self.metric_name = "accuracy"
        self.regular_validate = None

    def set_loss_func(self):
        self.loss = nn.CrossEntropyLoss()  # Accepts logits with labels

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def monitor_metric(self, output, label):
        class_pred = torch.argmax(output, dim=1).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(label, class_pred)
        return {self.metric_name: accuracy}

    def forward(self, images, text_vec, label=None):
        batch_size, _, _, _ = images.shape

        images = self.effnet(images)
        img_emb = self.linear1(images)
        text_emb = self.linear2(text_vec)


        full_emb = torch.cat([img_emb,text_emb], dim=1)

        if label is not None:
            out = self.arcface_head(full_emb, label)
            loss = self.loss(out, label)
            metric = self.monitor_metric(out, label)

            return out, loss, metric
        else:
            return full_emb, 0, {}

    def train_one_batch(self, images, text_vec, y_batch, device):
        self.optimizer.zero_grad()

        # Move data to device
        images, text_vec, y_batch = images.to(device), text_vec.to(device), y_batch.to(device)

        # Forwardpass
        out, loss, metric = self(images, text_vec, label=y_batch)
        # Calculate gradients
        loss.backward()
        # Backpropagate the gradients
        self.optimizer.step()

        wandb.log({"epoch": self.current_epoch, "loss": loss.item(), "metric": metric["accuracy"]}, step=self.current_train_step)

        return loss, metric

        # Method for training one single epoch

    def train_one_epoch(self, device):

        # Take to training mode
        self.train()

        # Define number of batches
        n_batches = len(self.train_loader)

        # Initiate tqdm object
        tk0 = tqdm(self.train_loader, total=n_batches)

        # For each batch ...
        for batch_no, (images, text_vec, y_batch) in enumerate(tk0):
            # ... Train model
            loss, metric = self.train_one_batch(images, text_vec, y_batch, device)

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="train", Batch_No=self.current_train_step, Batch_Loss=loss.item(), Batch_Metric=metric["accuracy"])

            self.current_train_step +=1

        tk0.close()

    def validate_all(self, device):
        # Set model to evalutation mode
        self.eval()

        # Define number of batches
        n_batches = len(self.valid_loader)

        # Initiate tqdm object
        tk0 = tqdm(self.valid_loader, total=n_batches)

        # Variables for storing total epoch errors
        total_loss = 0
        total_metric = 0

        # For each batch ...
        for batch_no, (images, text_vec, y_batch) in enumerate(tk0):
            # ... Calculate metrics on validation set without training
            loss, metric = self.validate_one_batch(images,text_vec, y_batch, device)

            # ... Cumulative sum up errors
            total_loss += loss.item()
            total_metric += metric["accuracy"]

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="Validation", Batch_No=batch_no, Batch_Loss=loss.item(),
                            Batch_Metric=metric["accuracy"])

        tk0.close()

    def validate_one_batch2(self, images, text_vec, y_batch, device):
        # Move validation batch to "device"
        images,text_vec, y_batch = images.to(device), text_vec.to(device), y_batch.to(device)

        with torch.no_grad():
            # Forwardpass
            # Forwardpass
            out, loss, metric = self(images, label=y_batch)

            return loss, metric

        # Main method for trianing all epochs

    def fit(self,
            train_dataset,
            valid_dataset=None,
            config={}):

        # Set model attributes
        self._set_model_attributes(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            config=config)

        wandb.watch(self, self.optimizer, log="all", log_freq=10)
        # For every epoch ...
        for epoch_no in range(self.n_epochs):
            # ... train the model
            self.train_one_epoch(device)  # Removed self.train_loader

            # ... Calculate loss and metric on validation
            if (self.valid_loader):
                if self.regular_validate:
                    self.validate_all(device)  # Removed self.valid_loader

            # ... Move scheduler if applicable
            if self.scheduler:
                self.scheduler.step()

            if (self.current_epoch+1) % self.f1_monitor_rate == 0:
                train_emb = self.predict(train_dataset, self.device, self.train_batch_size)
                train_dist_matrix = cdist(train_emb, train_emb, "cosine")
                train_best_threshold, train_threshold_scores = cluster(train_dist_matrix, [0.3], train_dataset) #np.arange(0.2,0.5,0.05)
                wandb.log({"epoch": self.current_epoch,
                           "train_f1": train_dataset.df.f1.mean(),
                           "train_best_threshold": train_best_threshold,
                           "train_threshold_scores": wandb.Table(
                               data=[[str(round(val, 2)) for val in train_threshold_scores.values()]],
                               columns=[str(round(key, 2)).replace(".", "_") for key in
                                        train_threshold_scores.keys()])}, step=self.current_train_step)

                if valid_dataset:
                    valid_emb = self.predict(valid_dataset, self.device, self.valid_batch_size)
                    valid_dist_matrix = cdist(valid_emb, valid_emb, "cosine")
                    val_best_threshold, val_threshold_scores = cluster(valid_dist_matrix, [0.3], valid_dataset)
                    wandb.log({"valid_f1": valid_dataset.df.f1.mean(),
                               "val_best_threshold": val_best_threshold,
                               "val_threshold_scores": wandb.Table(data=[[str(round(val, 2)) for val in val_threshold_scores.values()]], columns=[str(round(key, 2)).replace(".", "_") for key in val_threshold_scores.keys()])}, step=self.current_train_step)



            # Keep record of epoch_no
            self.current_epoch += 1

        # Helper function to set model attributes when .fit() is called
    def _set_model_attributes(self, train_dataset, valid_dataset, config):

        # Set number of epochs
        self.n_epochs = config["n_epochs"]
        self.LR = config["LR"]
        self.train_batch_size = config["train_batch_size"]
        self.valid_batch_size = config["valid_batch_size"]
        self.device = config["device"]
        self.f1_monitor_rate = config["f1_monitor_rate"]
        self.threshold = config["threshold"]
        self.regular_validate = config["regular_validate"]

        # Move model to "device"
        if next(self.parameters()).device != device:
            self.to(device)

        # Create train_loader from train_dataset
        if self.train_loader is None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True
            )
            # Set number of training batches
            self.train_n_batches = len(self.train_loader)

        # Create valid_loader from valid_dataset
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=self.valid_batch_size,
                    shuffle=False
                )
                # Set number of validation batches
                self.valid_n_batches = len(self.valid_loader)

        # Initiate Loss funciton
        if self.loss is None:
            self.set_loss_func()

        # Initiate Optimizer
        if self.optimizer is None:
            self.set_optimizer(self.LR)

        # Set Learning Rate Scheduler
        if self.scheduler is None:
            self.set_scheduler()  # Requires optimizer already created

    def predict_one_batch(self, images,text_vec, device):
        # Move data to device
        images, text_vec = images.to(device), text_vec.to(device)

        with torch.no_grad():
            embeddings, _, _ = self(images, text_vec)

        return embeddings

        # Method for unlabeled data prediction

    def predict(self, test_dataset, device, batch_size=16):

        # Check if the mode is at "device"
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        # Create data loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        # Set evaluation mode
        self.eval()

        # Variable to store predictions
        embeddings = []

        # Initiate tqdm object
        tk0 = tqdm(test_loader, total=len(test_loader))

        # For every test batch ...
        for batch_no, (images,text_vec, _) in enumerate(tk0):
            # Make predictions
            out = self.predict_one_batch(images,text_vec, device)

            # Move predictions to "cpu" device
            embeddings.append(out.cpu().detach().numpy())

            # Modify printing of tqdm
            tk0.set_postfix(stage="test")

        tk0.close()

        return np.concatenate(embeddings, axis=0)
