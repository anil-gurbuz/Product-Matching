from lib import *

# DATASET REQUIREMENTS
# 1. For training&Validation sets:  return X, y
# 2. For test sets:  return X

# MODEL CLASS REQUIREMENTS
# 1. Forward function: return out, loss, metric OR return out, 0, 0

class Base_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_loader = None
        self.valid_loader = None

        # Set via config
        self.train_batch_size = None
        self.valid_batch_size = None
        self.n_epochs = None
        self.validate_every_n_epoch = None
        self.device = None
        self.LR = None

        # Manually initialized -- MUST
        self.loss = None
        self.metric = None
        self.optimizer = None
        self.scheduler = None

        # Already initialized
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0



    # Modify this as appropriate for the model in use
    def forward(self, *args, **kwargs):
        # Takes data_batch as parameter -- below parts are based on this as well.

        # if label:
        #     return out, loss, metric
        # else:
        #     return out, 0, 0
        pass

    def set_optimizer(self):
        pass

    def set_scheduler(self):
        pass

    def set_extra_config(self, config):
        pass # Set extra configs here such as self.threshold = config["threshold"]

    # Helper function to set model attributes when .fit() is called
    def _set_model_attributes(self, train_dataset, valid_dataset, config):

        # Can pass directly in more parameters  using config
        # Setting configurations
        for attr in config.keys():
            self.__setattr__(attr, config[attr])

        # Move model to "device"
        if next(self.parameters()).device != device:
            self.to(device)

        # Create train_loader from train_dataset
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True
            )

        # Create valid_loader from valid_dataset
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=self.valid_batch_size,
                    shuffle=False
                )

        # Initiate Optimizer
        if self.optimizer is None:
            self.set_optimizer(self.LR)

        # Set Learning Rate Scheduler
        if self.scheduler is None:
            self.set_scheduler()  # Requires optimizer already created


    # Main method for trianing all epochs
    def fit(self,train_dataset, valid_dataset=None, config=None ):

        # Set model attributes
        self._set_model_attributes(train_dataset=train_dataset, valid_dataset=valid_dataset, config=config)

        wandb.watch(self, self.optimizer, log="all", log_freq=10)
        # For every epoch ...
        for _ in range(self.n_epochs):
            # ... train the model
            self.train_one_epoch()  # Removed self.train_loader

            # ... Calculate loss and metric on validation
            if self.valid_loader:
                if (self.current_epoch+1) % self.validate_every_n_epoch ==0 :
                    self.validate_all(valid_dataset) #  MODIFIED FOR SHOPEE ONLY -- REMOVE ARGUMENT HERE

            # ... Move scheduler if applicable
            if self.scheduler:
                self.scheduler.step()

            # Keep record of epoch_no
            self.current_epoch += 1

    # Method for training one single epoch
    def train_one_epoch(self):

        self.train()

        # Initiate tqdm object
        tk0 = tqdm(self.train_loader, total= len(self.train_loader))

        # Variables for storing total epoch errors
        loss_meter = AvgMeter(name="train_loss")
        metric_meter = AvgMeter(name="train_metric")

        # For each batch ...
        for batch_no, train_batch in enumerate(tk0):
            # ... Train model
            out, loss, metric  = self.train_one_batch(train_batch)

            loss_meter.update(loss.item()) ; metric_meter.update(metric)

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="train", Epoch_No=self.current_epoch, Batch_No=batch_no, Loss=loss.item(), Metric=metric)
            # ... Log to WANDB
            wandb.log({"epoch": self.current_epoch, "loss": loss.item(), "train_metric":metric} ,step=self.current_train_step)

            self.current_train_step +=1

        tk0.close()

        logging.info(loss_meter); logging.info(metric_meter)
        # Log epoch averages to wandb
        wandb.log({"epoch": self.current_epoch, "train_epoch_loss": loss_meter.avg,  "train_epoch_metric": metric_meter.avg}, step= self.current_train_step)



    def train_one_batch(self, train_batch):
        # Refresh gradients of the optimizer
        self.optimizer.zero_grad()

        for key, value in train_batch.items():
            train_batch[key] = value.to(self.device)

        # Forwardpass
        out, loss, metric = self(train_batch)

        # Calculate gradients
        loss.backward()
        # Backpropagate the gradients
        self.optimizer.step()

        # Move scheduler if exists
        if self.scheduler:
            self.scheduler.step()

        return out, loss, metric


    def validate_all(self):
        # Set model to evalutation mode
        self.eval()

        # Initiate tqdm object
        tk0 = tqdm(self.valid_loader, total=len(self.valid_loader))

        # Variables for storing total epoch errors
        loss_meter = AvgMeter(name="Valid_loss")
        metric_meter = AvgMeter(name="Valid_metric")

        # For each batch ...
        for batch_no, (valid_batch) in enumerate(tk0):

            out, loss, metric = self.validate_one_batch(valid_batch)

            loss_meter.update(loss.item()); metric_meter.update(metric)

            # ... Make tqdm print stats
            tk0.set_postfix(Stage="Validation", Epoch_No=self.current_epoch, Batch_No=batch_no, Loss=loss.item(), Metric=metric)
            # ... Log to WANDB

        tk0.close()

        logging.info(loss_meter); logging.info(metric_meter)
        # Log epoch averages to wandb
        wandb.log({"valid_loss": loss_meter.avg, "valid_metric": metric_meter.avg}, step=self.current_train_step)


    def validate_one_batch(self, valid_batch):

        # Move batch to device
        for key, value in valid_batch.items():
            valid_batch[key] = value.to(self.device)

        with torch.no_grad():
            # Forwardpass
            out, loss, metric = self(valid_batch)

        return out, loss, metrics

    # Method for unlabeled data prediction
    def predict(self, test_dataset, batch_size=16):

        # Check if the mode is at "device"
        if next(self.parameters()).device != self.device:
            self.to(self.device)

        # Create data loader
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Set evaluation mode
        self.eval()

        # Variable to store predictions
        preds = []

        # Initiate tqdm object
        tk0 = tqdm(test_loader, total=len(test_loader))

        # For every test batch ...
        for batch_no, (test_batch) in enumerate(tk0): # Test dataset doesn't return empty labels
            # Make predictions
            out = self.predict_one_batch(test_batch)

            # Move predictions to "cpu" device
            preds.append(out.cpu().detach().numpy())

            # Modify printing of tqdm
            tk0.set_postfix(stage="test")

        tk0.close()

        return np.concatenate(preds, axis=0)

    def predict_one_batch(self, test_batch):

        # Move batch to device
        for key, value in test_batch.items():
            test_batch[key] = value.to(self.device)

        with torch.no_grad():
            out, _, _ = self(test_batch)

        return out

    def snapshot(self, model_path, config):

        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None

        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None

        model_dict = {}
        model_dict["model_state_dict"] = self.state_dict()
        model_dict["optimizer_state_dict"] = opt_state_dict
        model_dict["scheduler_state_dict"] = sch_state_dict
        model_dict["config"] = config

        torch.save(model_dict, model_path)


def load_snapshot(model_path,  model_obj, device, optimizer_obj=None, scheduler_obj=None):

    checkpoint = torch.load(model_path, map_location=device) # Might need to add , map_location="cpu" or "gpu" depending where it is saved from

    model_obj.load_state_dict(checkpoint['model_state_dict'])

    if optimizer_obj:
        optimizer_obj.load_state_dict(checkpoint['optimizer_state_dict'])
        model_obj.optimizer = optimizer_obj

    if scheduler_obj:
        scheduler_obj.load_state_dict(checkpoint['scheduler_state_dict'])
        model_obj.scheduler = scheduler_obj

    config = checkpoint['config']
    for attr in config.keys():
        model_obj.__setattr__(attr, config[attr])

    model_obj.to(device)

    return model_obj


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




#
# # Tracker for epoch based metric values
# class Tracker():
#     def __init__(self):
#         self.train_epoch_meter = np.zeros(shape=(0, 0))
#         self.val_epoch_meter = np.zeros(shape=(0, 0))
#         self.test_epoch_meter = np.zeros(shape=(0, 0))
#
#     def add_to_train_meter(self, value):
#         self.train_epoch_meter = np.append(self.train_epoch_meter, value)
#
#     def add_to_val_meter(self, value):
#         self.val_epoch_meter = np.append(self.val_epoch_meter, value)
#
#     def add_to_test_meter(self, value):
#         self.test_epoch_meter = np.append(self.test_epoch_meter, value)
