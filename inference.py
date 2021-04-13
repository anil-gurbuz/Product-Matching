from Utils import *
from Model import image_embedder
from clustering import *

tfidf_dim=TEXT_VEC_SIZE
img_emb=256
text_emb=256
#TEXT_VEC_SIZE = TEXT_VEC_SIZE
threshold = 0.25
model_path = "data/tfidf_15000_256_256_11014.pth"

# Create test dataset
test_ds, _ = create_train_test(mode="inference")

# Load trained model
model = image_embedder(tfidf_dim=tfidf_dim,  img_emb=img_emb, text_emb=text_emb)
model.load_state_dict(torch.load(model_path))

model.device=device
# Generate embeddings and then submission file
embeddings = model.predict(test_ds) # If doesn't work, self.device attribute might be missing
cosine_find_matches_cupy(embeddings, test_ds.df.posting_id, threshold, create_submission=True)