from Utils import *
from Model import image_embedder
from clustering import *

img_embed_size = 512
text_embed_size = 512
TEXT_VEC_SIZE = 15000
threshold = 0.3
model_path = "data/Embedder_full_train_tfidf.pth"

# Create test dataset
test_ds, _ = create_train_test(mode="inference")

# Load trained model
model = image_embedder(tfidf_dim=TEXT_VEC_SIZE,  img_emb=img_embed_size, text_emb=text_embed_size)
model.load_state_dict(torch.load(model_path))

# Generate embeddings and then submission file
embeddings = model.predict(test_ds) # If doesn't work, self.device attribute might be missing
cosine_find_matches_cupy(embeddings, test_ds.df.posting_id, threshold, create_submission=True)