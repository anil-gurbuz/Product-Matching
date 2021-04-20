from lib import *


def get_transforms(img_size=IMAGE_SIZE):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize()
    ])


class ShopeeDataset(Dataset):
    def __init__(self, df, mode, transforms=get_transforms()):
        self.df = df.reset_index(drop=True)
        self.transform = transforms
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index,]
        try:
            label_group = torch.tensor(row.label_group)
        except (ValueError, AttributeError):
            label_group = torch.Tensor()


        image = cv2.imread(row.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = image["image"].astype(np.float32)
        image = image.transpose(2, 0, 1) # Turn into pytorch format # Batch, Channels, ...
        image = torch.tensor(image)

        return {"image":image,  "label":label_group}
