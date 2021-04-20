from Utils import *
from Model import image_embedder
from clustering import *
from Utils import get_text_embeddings

tfidf_dim=TEXT_VEC_SIZE
img_emb=256
#TEXT_VEC_SIZE = TEXT_VEC_SIZE
threshold = 0.25
model_path = "data/tfidf_15000_256_256_11014.pth"

# Create test dataset
test_ds, _ = create_train_test(mode="inference")

# Load trained model
model = image_embedder(img_emb=img_emb)
model.load_state_dict(torch.load(model_path))

model.device=device
# Generate embeddings and then submission file
img_emb = model.predict(test_ds) # If doesn't work, self.device attribute might be missing

text_emb = get_text_embeddings(test_ds.df, TEXT_VEC_SIZE)

cosine_find_matches_cupy(embeddings, test_ds.df.posting_id, threshold, create_submission=True)