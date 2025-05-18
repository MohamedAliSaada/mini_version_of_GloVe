# 🧠 MiniGloVe: Building Custom Static Word Embeddings

This project demonstrates how to create your own **GloVe-style static word embeddings** using a small dataset. It follows a simplified pipeline of GloVe (Global Vectors for Word Representation) and includes visualization and similarity scoring tools.

---

## 📂 Files Included

- `mini_version_of_GloVe.py`: The full script for training custom GloVe embeddings.
- `glove_custom_4d_4999W.txt`: Saved word vectors in GloVe format (4-dimensional, 4999 words).
- `how_to_use_the_text_file.ipynb`: A notebook demonstrating how to load and use the `.txt` file for further applications.

---

## 🧪 Dataset Used

We use text from:
- **SetFit/bbc-news**
- **fancyzhx/ag_news**

All are loaded via 🤗 Hugging Face `datasets`.

---

## 🧩 Workflow

1. **Load dataset and tokenize**  
   Clean, lowercase, and convert all sentences to token sequences.

2. **Build co-occurrence matrix**  
   Using a sliding window of size 2, count how often words co-occur with one another.

3. **Generate embeddings using SVD**  
   Apply Singular Value Decomposition on the co-occurrence matrix and extract the top components.

4. **Save to GloVe format**  
   Embeddings are saved as a text file:  



---

## 🧠 How to Use the Embeddings

### 1. Load the GloVe `.txt` file

```python
import numpy as np

def load_glove_embeddings(file_path):
 embeddings = {}
 with open(file_path, 'r', encoding='utf8') as f:
     for line in f:
         values = line.strip().split()
         word = values[0]
         vec = np.array(values[1:], dtype=np.float32)
         embeddings[word] = vec
 return embeddings

embeddings = load_glove_embeddings("glove_custom_4d_4999W.txt")


from sklearn.metrics.pairwise import cosine_similarity

def word_similarity(word1, word2, emb_dict):
    return cosine_similarity([emb_dict[word1]], [emb_dict[word2]])[0][0]


from tensorflow.keras.models import load_model
model = load_model("imdb.h5")  # or "imdb.keras"
