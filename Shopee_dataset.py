from lib import *

def get_transforms(img_size=256):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize()
    ])


class ShopeeDataset(Dataset):
    def __init__(self, df, mode, transforms=get_transforms(), tokenizer=None):
        self.df = df.reset_index(drop=True)
        self.transform = transforms

        ## NOT NEEDED ATM ###
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index,]

        text = row.title
        image = cv2.imread(row.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res0 = self.transform(image=image)
        image0 = res0["image"].astype(np.float32)
        image = image0.transpose(2, 0, 1) # Turn into pytorch format # Batch, Channels, ...

        if self.tokenizer:
            text = self.tokenizer(text, padding="max_length", truncation=True, max_length=16, retun_tensorts="pt")
            input_ids = text["input_ids"][0]
            attention_mask = text["attention_mask"][0] # MIGHT NEED TO TURN INTO TENSOR
        else:
            input_ids = torch.Tensor()
            attention_mask = torch.Tensor()

        if self.mode == "test":
            return torch.tensor(image), input_ids, attention_mask
        else:
            return torch.tensor(image), input_ids, attention_mask, torch.tensor(row.label_group)









