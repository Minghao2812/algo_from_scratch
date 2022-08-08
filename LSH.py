"""Locality Sensitive Hashing (LSH) for similarity search with Shingling & MinHash.
"""
import random


class LSH:
    def __init__(self) -> None:
        pass

    def create_vocab(self, *docs):
        """Create vocab including all documents.
        Parameters
        ----------
        *args :
            Documents to make up the vocab.
        """
        self.vocab = set()
        for doc in docs:
            self.vocab = self.vocab.union(doc)
        self.vocab_size = len(self.vocab)
        return self.vocab

    def shingling(self, text: 'str', window: 'int' = 2):
        """Split the text character by character with a fix window length.
        It's the same as n-gram.

        Parameters
        ----------
        text :
            The text to be done with shingling.
        window :
            The shingling window size.
        """
        shingling = []
        for i in range(len(text) - window + 1):
            shingling.append(text[i:i + window])
        return list(set(shingling))

    def onehot(self, doc: 'str'):
        """
        Parameters
        ----------
        vocab :
            The vocab.
        doc :
            The doc to create one-hot vector.

        Return
        ------
        One-hot vector for the doc. The vector has the same length as vocab (|V|).
        """
        onehot = []
        for word in self.vocab:
            if word in doc:
                onehot.append(1)
            else:
                onehot.append(0)
        return onehot

    def create_hash_idx(self, digit: 'int' = 20) -> 'list(list(int))':
        hash_idx_list = []
        for _ in range(digit):
            hash_num = list(range(self.vocab_size))
            random.shuffle(hash_num)
            hash_idx_list.append(hash_num)
        return hash_idx_list

    def minhash(self, onehot, hash_idx_list: 'list(list(int))'):
        signature = []
        for hash_idx in range(len(hash_idx_list)):
            for num in range(self.vocab_size):
                idx = hash_idx.index(num)
                if onehot[idx] == 1:
                    signature.append(idx)
                    break

        return signature

    def banding(self, signature, k: 'int') -> 'list(list(int))':
        """Cut the signature vector into k pieces.
        Parameters
        ----------
        signature :
            The signature vector of a doc's one-hot vector.
        k :
            Number of pieces.
        """
        assert len(signature
                   ) % k == 0, 'Length of signature should be divisible by b!'
        banding = []
        window = len(signature) / k
        for i in range(len(signature)):
            banding.append(signature[i:i + window])
        return banding

    # def jaccard(self, a, b):
    #     a = set(a)
    #     b = set(b)
    #     return len(a.intersection(b)) / len(a.union(b))

    def compare(self, *args):
        banding1, banding2 = args
        for sub1, sub2 in zip(banding1, banding2):
            if sub1 == sub2:
                return True


if __name__ == '__main__':
    text1 = """You are not smart."""
    text2 = """You are not very smart."""
    text3 = """You are stupid."""

    model = LSH()

    shingling1 = model.shingling(text1)
    shingling2 = model.shingling(text2)
    shingling3 = model.shingling(text3)
    print('Shingling1: ', shingling1)
    print('Shingling2: ', shingling2)
    print('Shingling3: ', shingling3)

    model.create_vocab(shingling1, shingling2, shingling3)
    print('Vocab: ', model.vocab)
    print('Vocab size: ', model.vocab_size)

    onehot1 = model.onehot(shingling1)
    onehot2 = model.onehot(shingling2)
    onehot3 = model.onehot(shingling3)
    print('One-hot1: ', onehot1)
    print('One-hot2: ', onehot2)
    print('One-hot3: ', onehot3)

    hash_idx_list = model.create_hash_idx(20)
    print('Hash indices: ', hash_idx_list)
    print('Hash indices size: ', model.vocab_size * len(hash_idx_list))

    # sig1 = create_signature(onehot1, hash_num, len(vocab))
    # sig2 = create_signature(onehot2, hash_num, len(vocab))
    # sig3 = create_signature(onehot3, hash_num, len(vocab))
    # print('Signature1: ', sig1)
    # print('Signature2: ', sig2)
    # print('Signature3: ', sig3)