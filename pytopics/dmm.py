import numpy as np
import pandas as pd
import numba
import tqdm
import scipy


class GSDMM:
    """
    Gibbs Sampling algorithm for the Dirichlet Multinomial Mixture model
    http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

    Parameters
    ----------
    n_components: int, number of topics
    alpha: float, alpha parameter of Dirichlet prior on topics
    beta: float, beta parameter of Dirichlet prior on mixture components
    max_iter: int, number of max iterations of sampling
    """

    def __init__(self, n_components, alpha=1.0, beta=1.0, max_iter=10):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def fit(self, X, min_changed_stopping=0, verbose=False):
        text_vector_indices = self._get_text_vector_indices(X)
        topic_document_counts, topic_sizes, topic_word_counts, document_topics = self._make_topics(text_vector_indices, self.n_components)
        self.topic_document_counts = topic_document_counts
        self.topic_sizes = topic_sizes
        self.topic_word_counts = topic_word_counts
        self.components_ = topic_word_counts / topic_word_counts.sum(axis=1)[:, np.newaxis]
        _iter = range(self.max_iter)
        if verbose:
            _iter = tqdm.tqdm(_iter)
        for i in _iter:
            prev_document_topics = np.copy(document_topics)
            self.step(text_vector_indices, self.alpha, self.beta, topic_document_counts, topic_sizes, topic_word_counts, document_topics)
            n_changed = np.sum(prev_document_topics != document_topics)
            if n_changed <= min_changed_stopping:
                print('Converged at {}th iteration'.format(i))
                break
        return self

    def transform(self, X):
        text_vector_indices = self._get_text_vector_indices(X)
        def sampling_1d(doc):
            return self._sampling_distribution(doc, self.alpha, self.beta, X.shape[0], self.topic_document_counts, self.topic_sizes, self.topic_word_counts)

        return np.apply_along_axis(sampling_1d, axis=1, arr=text_vector_indices)

    @classmethod
    def _get_text_vector_indices(cls, text_vectors):
        """
        Convert sparse binary count vectorizer's matrix into dense matrix
        doc_word_indices[i, j] = w
        means that in document *i* word *j* has positon *w* in dictionary
        """
        n_documents = text_vectors.shape[0]
        x_indices, y_indices, __ = scipy.sparse.find(text_vectors)
        max_doc_length = pd.DataFrame({'y': y_indices, 'x': x_indices}).groupby('x').agg('count').max()[0]
        doc_word_indices = -np.ones((n_documents, max_doc_length + 1), dtype='int32')
        for i in range(n_documents):
            word_indices = y_indices[x_indices == i]
            for j in range(len(word_indices)):
                doc_word_indices[i, j] = word_indices[j]
        return doc_word_indices

    @staticmethod
    @numba.jit(nopython=True)
    def _count_words_in_topics(document_topics, text_vector_indices, topic_word_counts):
        for text in range(text_vector_indices.shape[0]):
            for i in text_vector_indices[text]:
                if i < 0:
                    break
                topic_word_counts[document_topics[text], i] += 1
        return topic_word_counts

    @classmethod
    def _make_topics(cls, text_vector_indices, num_topics):
        num_texts, __ = text_vector_indices.shape
        num_words = text_vector_indices.max()
        topic_document_counts = np.zeros((num_topics,)) # m_z
        topic_sizes = np.zeros((num_topics,)) # n_z
        topic_word_counts = np.zeros((num_topics, num_words+1)) # n^w_z

        text_lenghts = np.array((text_vector_indices >= 0).sum(axis=1))

        document_topics = np.random.randint(0, num_topics, size=num_texts)
        topic_word_counts = cls._count_words_in_topics(document_topics, text_vector_indices, topic_word_counts)

        document_topics_ohe = np.zeros((num_texts, num_topics))
        document_topics_ohe[np.arange(num_texts), document_topics] = 1

        topic_document_counts += document_topics_ohe.sum(axis=0)

        topic_sizes = (text_lenghts[:, np.newaxis] * document_topics_ohe).sum(axis=0)
        return topic_document_counts, topic_sizes, topic_word_counts, document_topics

    @staticmethod
    @numba.jit(nopython=True)
    def _sampling_distribution(doc, alpha, beta, D, topic_document_counts, topic_sizes, topic_word_counts):
        """
        sampling distribution for document doc given m_z, n_z, n^w_z
        formula (3) from paper
        """
        K, V = topic_word_counts.shape
        m_z, n_z, n_z_w = topic_document_counts, topic_sizes, topic_word_counts

        log_p = np.zeros((K,))

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = np.log(m_z[z] + alpha)
        #  lN2 = np.log(D - 1 + K*alpha)
        #  lN2 = np.log(product(n_z_w[w] + beta)) = sum(np.log(n_z_w[w] + beta))
        #  lD2 = np.log(product(n_z[d] + V*beta + i -1)) = sum(np.log(n_z[d] + V*beta + i -1))

        lD1 = np.log(D - 1 + K * alpha)
        for label in range(K):
            lN1 = np.log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            doc_size = 0
            for word in doc:
                if word < 0:
                    break
                lN2 += np.log(n_z_w[label, word] + beta)
                doc_size += 1
            for j in range(1, doc_size +1):
                lD2 += np.log(n_z[label] + V * beta + j - 1)
            log_p[label] = lN1 - lD1 + lN2 - lD2

        log_p = log_p - log_p.max() / 2
        p = np.exp(log_p)
        # normalize the probability vector
        pnorm = p.sum()
        pnorm = pnorm if pnorm>0 else 1
        return p / pnorm

    @staticmethod
    @numba.jit(nopython=True)
    def _update_topic(doc, topic, topic_sizes, topic_document_counts, topic_word_counts, update_int):
        topic_document_counts[topic] += update_int
        for w in doc:
            if w < 0:
                break
            topic_sizes[topic] += update_int
            topic_word_counts[topic][w] += update_int

    @classmethod
    def step(cls, text_vector_indices, alpha, beta, topic_document_counts, topic_sizes, topic_word_counts, document_topics):
        D, maxsize = text_vector_indices.shape
        K, V = topic_word_counts.shape
        for d in range(D):
            doc = text_vector_indices[d]
            # update old
            previous_cluster = document_topics[d]
            cls._update_topic(doc, previous_cluster, topic_sizes, topic_document_counts, topic_word_counts, -1)
            # sample
            p = cls._sampling_distribution(doc, alpha, beta, D, topic_document_counts, topic_sizes, topic_word_counts)
            new_cluster = np.argmax(np.random.multinomial(1, p))
            document_topics[d] = new_cluster
            # update new
            cls._update_topic(doc, new_cluster, topic_sizes, topic_document_counts, topic_word_counts, 1)