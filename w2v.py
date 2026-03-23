import numpy as np
import re
from tqdm import tqdm
from collections import Counter


class Word2Vec:
    def __init__(
            self, 
            corpus: str, 
            vec_size: int = 50, 
            window_size: int = 5, 
            seed: int = 42, 
            batch_size: int = 100,
            num_epochs: int = 20,
            lr: float = 0.1,
            min_count: int = 5
        ) -> None:
        """
        CBOW Word2Vec with full softmax and <UNK> handling for rare words.
 
        Args:
            corpus: raw text string to train on
            vec_size: dimensionality of word embeddings
            window_size: number of context words on each side of the target
            seed: random seed for reproducibility
            batch_size: number of training samples per gradient update
            num_epochs: number of full passes over the training data
            lr: initial learning rate (linearly decayed during fit)
            min_count: words appearing fewer than this many times become <UNK>
        """

        self.id2word = {}
        self.word2id = {}
        self.corpus_ids = []

        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        words  = pattern.findall(corpus.lower())
        counts  = Counter(words)

        UNK = '<UNK>'
        unk_id = 0
        self.id2word[unk_id] = UNK
        self.word2id[UNK] = unk_id

        for word, count in counts.items():
            word_id = self.word2id.get(word)
                
            if word_id is None and count >= min_count:
                word_id = len(self.id2word)
                self.id2word[word_id] = word
                self.word2id[word] = word_id

        self.corpus_ids = np.array([self.word2id.get(word, unk_id) for word in words])    

        self.vocab_size = len(self.id2word)
        self.vec_size = vec_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        np.random.seed(seed)
        self.Emb = np.random.randn(self.vocab_size, vec_size) * 0.1
        self.W = np.random.randn(vec_size, self.vocab_size) * 0.1

        self._generate_training_data()
    

    def _generate_training_data(self) -> None:
        """
        Build (context, target) pairs from the corpus.
 
        <UNK> tokens are skipped as targets since they represent many different
        rare words and would add noise. They are kept as context words so that
        window positions around them remain correct.
        """

        x = []
        y = []
        unk_id = self.word2id['<UNK>']

        for i in range(len(self.corpus_ids)):
            if self.corpus_ids[i] == unk_id:
                continue

            start_pos = max(0, i - self.window_size)
            end_pos = min(len(self.corpus_ids), i + self.window_size + 1)

            context = [self.corpus_ids[j] for j in range(start_pos, end_pos) if j != i]
            x.append(context)
            y.append(self.corpus_ids[i])

        self.x = x
        self.y = np.array(y, dtype=np.int32)
    

    def _softmax(self, X: np.ndarray) -> np.ndarray:
        X = X - np.max(X, axis=1, keepdims=True)
        exp = np.exp(X)
        
        return exp / np.sum(exp, axis=1, keepdims=True)


    def _forward(self, X: list[list[int]]) -> dict[str, np.ndarray]:
        """
        Forward pass for a batch of context windows.
 
        Steps:
            1. Average the context embeddings to get the hidden layer (CBOW mean)
            2. Project hidden -> output space via W
            3. Apply softmax to get a probability distribution over the vocab
 
        Args:
            X: list of context id lists, one per sample
 
        Returns:
            cache dict with keys: hidden, output, prob, X
        """

        hidden = np.array([self.Emb[cur_x].mean(axis=0) for cur_x in X])
        output = hidden @ self.W
        prob = self._softmax(output)

        return {'hidden': hidden, 'output': output, 'prob': prob, 'X': X}
    

    def _backward(self, cache: dict[str, np.ndarray], Y: np.ndarray) -> None:
        """
        Backward pass using the cross-entropy + softmax combined gradient.
 
        The gradient of cross-entropy loss w.r.t. the softmax input simplifies
        to (prob - one_hot(Y)), which avoids computing the Jacobian explicitly.
        Updates Emb and W in-place via SGD.
 
        Args:
            cache: output of _forward for the current batch
            Y: target word ids, shape (batch,)
        """

        cur_batch_size = len(Y)

        error = cache['prob']
        error[np.arange(cur_batch_size), Y] -= 1
        error /= cur_batch_size

        dw2 = cache['hidden'].T @ error

        dh = error @ self.W.T
        for i, cur_x in enumerate(cache['X']):
            grad = dh[i] / len(cur_x)
            np.add.at(self.Emb, cur_x, -self.lr * grad)

        self.W -= self.lr * dw2


    def _cross_entropy(self, z: np.ndarray) -> float:
        '''
        Mean cross-entropy loss over a batch.
        '''
        return -np.sum(np.log(z + 1e-7)) / len(z)
    

    def _get_batch(self, shuffle: bool = True) -> tuple[list[list[int]], np.ndarray]:
        """
        Yield (X, Y) mini-batches from the training data.
 
        Args:
            shuffle: if True, randomize sample order each call
 
        Yields:
            X: list of context id lists for the batch
            Y: target word ids, shape (batch_size,)
        """

        n_samples = len(self.x)

        indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)

        for i in range(0, n_samples, self.batch_size):
            batch_idx = indices[i: i + self.batch_size]
            X_batch = [self.x[j] for j in batch_idx]
            Y_batch = self.y[batch_idx]
            yield X_batch, Y_batch


    def fit(self) -> None:
        """
        Train the model for num_epochs passes over the data.
        Learning rate is linearly decayed across epochs.
        """

        initial_lr = self.lr

        for epoch in range(self.num_epochs):
            loss = 0.0
            num_batches = 0

            progress = epoch / self.num_epochs
            self.lr = max(initial_lr * (1.0 - progress), 0.0001)

            for X, Y in self._get_batch():
                cur_batch_size = len(Y)
                cache = self._forward(X)
                probs = cache['prob'][np.arange(cur_batch_size), Y]
                loss += self._cross_entropy(probs)
                self._backward(cache, Y)
                num_batches += 1

            print(f'Epoch: {epoch} | Avg Loss: {loss / num_batches}')


    def predict(self, word: str) -> np.ndarray:
        """
        Return the embedding vector for a word.
        Falls back to the <UNK> vector for out-of-vocabulary words.
 
        Args:
            word: input word
 
        Returns:
            embedding vector of shape (vec_size,)
        """

        ind = self.word2id.get(word)
        if ind is None:
            return self.Emb[self.word2id['<UNK>']]

        return self.Emb[ind, :]
    

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        """
        Find the topn most similar words by cosine similarity.
        <UNK> is excluded from results as it is not a meaningful neighbor.
 
        Args:
            word: query word
            topn: number of results to return
 
        Returns:
            list of (word, similarity) tuples sorted by descending similarity
        """

        vec = self.predict(word)
        norm_vec = vec / np.linalg.norm(vec).clip(min=1e-8)

        norms = np.linalg.norm(self.Emb, axis=1, keepdims=True).clip(min=1e-8)
        all_vecs = self.Emb / norms

        sims = all_vecs @ norm_vec
        top_ids = np.argsort(-sims)

        results = []
        for i in top_ids:
            w = self.id2word[i]
            if w != word and w != '<UNK>':
                results.append((w, float(sims[i])))
            if len(results) == topn:
                break

        return results