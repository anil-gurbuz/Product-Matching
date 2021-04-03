from lib import *
from ArcFace import ArcFace
from Base_Trainer import Base_model
from Base_Trainer import Tracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LR = 1e-4


class image_embedder(Base_model):
    def __init__(self, embed_size=128, out_classes=11014):
        super().__init__()

        try:
            self.effnet = EfficientNet.from_name("efficientnet-b3")
            self.effnet.load_state_dict(torch.load("data/efficientnet-b3-5fb5a3c3.pth"))
        except:
            self.effnet = EfficientNet.from_pretrained(
                'data\efficientnet-b3-5fb5a3c3.pth')  # Might try without last 2 layers -- basically with extracting the features

        self.linear = nn.Linear(1000, embed_size)
        self.arcface_head = ArcFace(embed_size, out_classes)


        self.optimizer = None
        self.metric_name = "accuracy"


    def set_loss_func(self):
        self.loss = nn.CrossEntropyLoss()  # Accepts logits with labels

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def monitor_metric(self, output, label):
        class_pred = torch.argmax(output, dim=1).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(label, class_pred)
        return {self.metric_name : accuracy}


    def forward(self, images, input_ids=None, attention_mask=None, label=None):
        batch_size, _, _, _ = images.shape

        images = self.effnet(images)
        embedding = self.linear(images)

        if label is not None:
            out = self.arcface_head(embedding, label)
            loss = self.loss(out, label)
            metric = self.monitor_metric(out, label)

            return out, loss, metric
        else:
            return embedding, 0, {}



    def train_one_batch(self, images, input_ids, attention_mask, y_batch, device):
        self.optimizer.zero_grad()

        # Move data to device
        images, input_ids, attention_mask, y_batch =  images.to(device), input_ids.to(device), attention_mask.to(device), y_batch.to(device)

        # Forwardpass
        out, loss, metric = self(images,label=y_batch)
        # Calculate gradients
        loss.backward()
        # Backpropagate the gradients
        self.optimizer.step()

        return loss, metric

        # Method for training one single epoch
    def train_one_epoch(self, device, trackers):

        # Take to training mode
        self.train()

        # Define number of batches
        n_batches = len(self.train_loader)

        # Initiate tqdm object
        tk0 = tqdm(self.train_loader, total=n_batches)

        # Variables for storing total epoch errors
        total_loss = 0
        total_metric = 0

        # For each batch ...
        for batch_no, (images, input_id, attention, y_batch) in enumerate(tk0):
            # ... Train model
            loss, metric = self.train_one_batch(images, input_id, attention, y_batch, device)


            # ... Cumulative sum up errors
            total_loss += loss.item()
            total_metric += metric["accuracy"]

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="train", Batch_No=batch_no, Batch_Loss=loss.item(), Batch_Metric=metric["accuracy"])

        tk0.close()

        # Save loss and metric value for train
        trackers["loss"].add_to_train_meter(total_loss / n_batches)
        trackers[self.metric_name].add_to_train_meter(total_metric / n_batches)

        return trackers

    def validate_all(self, device, trackers):
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
        for batch_no, (images, input_id, attention, y_batch) in enumerate(tk0):
            # ... Calculate metrics on validation set without training
            loss, metric = self.validate_one_batch(images, input_id, attention, y_batch, device)

            # ... Cumulative sum up errors
            total_loss += loss.item()
            total_metric += metric["accuracy"]

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="Validation", Batch_No=batch_no, Batch_Loss=loss.item(), Batch_Metric=metric["accuracy"])

        tk0.close()
        # Save loss and metric value for validation
        trackers["loss"].add_to_val_meter(total_loss / n_batches)
        trackers[self.metric_name].add_to_val_meter(total_metric / n_batches)

        return trackers

    def validate_one_batch(self, images, input_ids, attention_mask, y_batch, device):
        # Move validation batch to "device"
        images, input_ids, attention_mask, y_batch = images.to(device), input_ids.to(device), attention_mask.to(device), y_batch.to(device)

        with torch.no_grad():
            # Forwardpass
            # Forwardpass
            out, loss, metric = self(images, label=y_batch)

            return loss, metric


        # Main method for trianing all epochs
    def fit(self,
            train_dataset,
            valid_dataset=None,
            device="cuda",
            n_epochs=10,
            train_batch_size=16,
            valid_batch_size=16,
            lr = 1e-4
            ):

        # Set model attributes
        self._set_model_attributes(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            device=device,
            n_epochs=n_epochs,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            lr= lr
        )

        # Initialise trackers for training and validation tracking
        trackers = {"loss": Tracker(), self.metric_name: Tracker()}

        wandb.watch(self, self.optimizer, log="all", log_freq=10)
        # For every epoch ...
        for epoch_no in range(n_epochs):
            # ... train the model
            trackers = self.train_one_epoch(device, trackers)  # Removed self.train_loader

            wandb.log({"epoch":self.current_epoch, "loss":trackers["loss"]}, step=0)

            # ... Calculate loss and metric on validation
            if self.valid_loader:
                trackers = self.validate_all(device, trackers)  # Removed self.valid_loader

            # ... Move scheduler if applicable
            if self.scheduler:
                self.scheduler.step()

            # Keep record of epoch_no
            self.current_epoch += 1

        return trackers

        # Helper function to set model attributes when .fit() is called
    def _set_model_attributes(self, train_dataset, valid_dataset, device, n_epochs, train_batch_size,
                                  valid_batch_size, lr):

        # Set number of epochs
        self.n_epochs = n_epochs

        # Move model to "device"
        if next(self.parameters()).device != device:
            self.to(device)

        # Create train_loader from train_dataset
        if self.train_loader is None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True
            )
            # Set number of training batches
            self.train_n_batches = len(self.train_loader)

        # Create valid_loader from valid_dataset
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=valid_batch_size,
                    shuffle=False
                )
                # Set number of validation batches
                self.valid_n_batches = len(self.valid_loader)

        # Initiate Loss funciton
        if self.loss is None:
            self.set_loss_func()

        # Initiate Optimizer
        if self.optimizer is None:
            self.set_optimizer(lr)

        # Set Learning Rate Scheduler
        if self.scheduler is None:
            self.set_scheduler()  # Requires optimizer already created



    def predict_one_batch(self, images, input_ids, attention_mask, device):
        # Move data to device
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

        with torch.no_grad():
            embeddings, _, _ = self(images, input_ids, attention_mask)

        return embeddings



        # Method for unlabeled data prediction
    def predict(self, test_dataset, device, batch_size=16):

        # Check if the mode is at "device"
        if next(self.parameters()).device != device:
            self.to(device)

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
        for batch_no, (images, input_ids, attention_mask) in enumerate(tk0):
            # Make predictions
            out = self.predict_one_batch(images, input_ids, attention_mask, device)

            # Move predictions to "cpu" device
            embeddings.append(out.cpu().detach().numpy())

            # Modify printing of tqdm
            tk0.set_postfix(stage="test")

        tk0.close()

        return np.concatenate(embeddings, axis=0)
