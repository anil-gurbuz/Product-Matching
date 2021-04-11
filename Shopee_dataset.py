from lib import *


def get_transforms(img_size=256):
    return albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize()
    ])

# Function to get our text title embeddings
def get_text_embeddings(titles, max_features = TEXT_VEC_SIZE):

    vectorizer = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    vectorizer = vectorizer.fit(pd.read_csv(data_folder + "/train.csv"))
    text_embeddings = vectorizer.transform(titles)
    del vectorizer
    return text_embeddings

class ShopeeDataset(Dataset):
    def __init__(self, df, mode, transforms=get_transforms()):
        self.df = df.reset_index(drop=True)
        self.transform = transforms
        self.text_vec = get_text_embeddings(df["title"])
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index,]
        try:
            label_group = torch.tensor(row.label_group)
        except ValueError:
            label_group = None

        text_vec = self.text_vec[index,]
        text_vec = torch.tensor(np.squeeze(text_vec.toarray().astype(np.float32)))

        image = cv2.imread(row.file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = image["image"].astype(np.float32)
        image = image.transpose(2, 0, 1) # Turn into pytorch format # Batch, Channels, ...
        image = torch.tensor(image)

        return {"image":image, "text_vec":text_vec, "label":label_group}
