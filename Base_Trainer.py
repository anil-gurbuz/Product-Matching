from lib import *

# MIGHT WRITE A SUMMARY ABOUT WHAT EACH FUNCTION DOES. FOR EXAMPLE: TRAIN_ONE_BATCH -MOVES DATA TO DEVICE, FORWARDPASS ...
class Base_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_loader = None
        self.valid_loader = None

        self.train_n_batches = None
        self.val_n_batches = None
        self.n_epochs = None
        self.device = None
        self.LR = None
        self.baby_sit = None

        self.optimizer = None
        self.scheduler = None
        self.loss = None

        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0

        self.metric_name = None      # Have to set this
        # self.metrics["train"] = {}
        # self.metrics["valid"] = {}
        # self.metrics["test"] = {}

    # Modify this as appropriate for the model in use
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

        # Modify this to choose a loss function to use in training

    def set_loss_func(self, *args, **kwargs):
        return

    # Modify this to select from torch.optim..... classes
    def set_optimizer(self, *args, **kwargs):
        return

    # Modify this to select from torch.optim.lr_scheduler.... classes
    def set_scheduler(self, *args, **kwargs):
        return

    # Modify this to choose a metric function to use in training -- Will accept target and pred
    def monitor_metric(self, *args, **kwargs):

        # self.metric_name = ...
        return

    # Helper function to set model attributes when .fit() is called
    def _set_model_attributes(self, train_dataset, valid_dataset, device, n_epochs, train_batch_size, valid_batch_size):

        # Set number of epochs
        self.n_epochs = n_epochs

        # Move model to "device"
        if next(self.parameters()).device != device:
            self.to(device)

        # Create train_loader from train_dataset
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True
            )
            # Set number of training batches
            self.train_n_batches = len(self.train_loader)

        # Create valid_loader from valid_dataset
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=valid_batch_size,
                    shuffle=False
                )
                # Set number of validation batches
                self.valid_n_batches = len(self.valid_loader)

        # Initiate Optimizer
        if self.optimizer is None:
            self.optimizer = self.set_optimizer()

        # Set Learning Rate Scheduler
        if self.scheduler is None:
            self.scheduler = self.set_scheduler()  # Requires optimizer already created

    # Main method for trianing all epochs
    def fit(self,
            train_dataset,
            valid_dataset=None,
            device="cuda",
            n_epochs=10,
            train_batch_size=16,
            valid_batch_size=16
            ):

        # Set model attributes
        self._set_model_attributes(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            device=device,
            n_epochs=n_epochs,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size
        )

        # Initialise trackers for training and validation tracking
        trackers = {"loss": Tracker(), self.metric_name: Tracker()}

        # For every epoch ...
        for _ in range(n_epochs):
            # ... train the model
            trackers = self.train_one_epoch(device, trackers)  # Removed self.train_loader

            # ... Calculate loss and metric on validation
            if self.valid_loader:
                trackers = self.validate_all(device, trackers)  # Removed self.valid_loader

            # ... Move scheduler if applicable
            if self.scheduler:
                self.scheduler.step()

            # Keep record of epoch_no
            self.current_epoch += 1

    # Method for training one single epoch
    def train_one_epoch(self, device, trackers):
        # Set model to trianing mode
        global metric
        self.train()
        # Define number of batches
        n_batches = len(self.train_loader)
        # Initiate tqdm object
        tk0 = tqdm(self.train_loader, total=n_batches)

        # Variables for storing total epoch errors
        total_loss = 0
        total_metric = 0

        # For each batch ...
        for batch_no, training_batch in enumerate(tk0):
            # ... Train model
            loss, metric = self.train_one_batch(training_batch, device)

            # ... Cumulative sum up errors
            total_loss += loss.item()
            total_metric += metric

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="train", Batch_No=batch_no, Batch_Loss=loss, Batch_Metric=metric)

        tk0.close()

        # Save loss and metric value for train
        trackers["loss"].add_to_train_meter(total_loss / n_batches)
        trackers[self.metric_name].add_to_train_meter(total_metric / n_batches)

        return trackers

    def train_one_batch(self, training_batch, device):

        # Refresh gradients of the optimizer
        self.optimizer.zero_grad()

        # Move data to "device"
        for key, value in training_batch.items():
            training_batch[key] = value.to(device)

        # Forwardpass
        predictions = self(**training_batch)

        # Calculate loss keeping gradients--- Designed for reduced loss of the batch
        loss = self.loss(predictions, **training_batch[1])  # Check this might not be correct

        # Calculate gradients
        loss.backward()
        # Backpropagate the gradients
        self.optimizer.step()

        # Move scheduler if exists
        if self.scheduler:
            self.scheduler.step()

        # Calculate training metric for the current batch
        metric = create_metric(predictions, **training_batch[1])

        return loss, metric

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
        for batch_no, valid_batch in enumerate(tk0):
            # ... Calculate metrics on validation set without training
            loss, metric = self.validate_one_batch(valid_batch, device)

            # ... Cumulative sum up errors
            total_loss += loss.item()
            total_metric += metric

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="Validation", Batch_No=batch_no, Batch_Loss=loss, Batch_Metric=metric)

        tk0.close()
        # Save loss and metric value for train
        trackers["loss"].add_to_val_meter(total_loss / n_batches)
        trackers[self.metric_name].add_to_val_meter(metric / n_batches)

        return trackers

    def validate_one_batch(self, valid_batch, device):
        # Move validation batch to "device"
        for key, value in valid_batch.items():
            valid_batch[key] = value.to(device)

        with torch.no_grad():
            # Forwardpass
            predictions = self(**valid_batch)

            # Designed for reduced loss of the batch
            loss = loss_func(predictions, **valid_batch[1])  # Check this might not be correct
            metric = create_metric(predictions, **valid_batch[1])

        return loss, metrics

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
        preds = []

        # Initiate tqdm object
        tk0 = tqdm(test_loader, total=len(test_loader))

        # For every test batch ...
        for batch_no, test_batch in enumerate(tk0):
            # Make predictions
            out = self.predict_one_batch(test_batch, device)

            # Move predictions to "cpu" device
            preds.append(out.cpu().detach().numpy())

            # Modify printing of tqdm
            tk0.set_postfix(stage="test")

        tk0.close()

        return preds

    def predict_one_batch(self, test_batch, device):
        for key, value in test_batch.items():
            test_batch[key] = value.to(device)

        with torch.no_grad():
            out = self(**test_batch)

        return out

    def save(self, model_path):
        model_state_dict = self.state_dict()

        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None

        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None

        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch

        torch.save(model_dict, model_path)

    def load(self, model_path, device="cuda"):

        if next(self.parameters()).device != device:
            self.to(device)

        model_dict = torch.load(model_path)

        self.load_state_dict(model_dict["state_dict"])

        # Might need to load optimizer and scheduler ??


# Tracker for epoch based metric values
class Tracker():
    def __init__(self):
        self.train_epoch_meter = np.zeros(shape=(0, 0))
        self.val_epoch_meter = np.zeros(shape=(0, 0))
        self.test_epoch_meter = np.zeros(shape=(0, 0))

    def add_to_train_meter(self, value):
        self.train_epoch_meter = np.append(self.train_epoch_meter, value)

    def add_to_val_meter(self, value):
        self.val_epoch_meter = np.append(self.val_epoch_meter, value)

    def add_to_test_meter(self, value):
        self.test_epoch_meter = np.append(self.test_epoch_meter, value)

# MIGHT CHANGE TRACKER WITH THIS
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text






