"""Locality Sensitive Hashing (LSH) for similarity search with Shingling & MinHash.
"""
import random


class LSH:
    def __init__(self) -> None:
        pass

    def create_vocab(self, *docs) -> 'set(str)':
        """Create vocab including all documents.

        Parameters
        ----------
        *args :
            Documents to make up the vocab.
        
        Return
        ------
        The vocab set.
        """
        self.vocab = set()
        for doc in docs:
            self.vocab = self.vocab.union(doc)
        self.vocab_size = len(self.vocab)
        return self.vocab

    def shingling(self, text: 'str', window: 'int' = 2) -> 'list(str)':
        """Split the text character by character with a fix window length.
        It's the same as n-gram.

        Parameters
        ----------
        text :
            The text to be done with shingling.
        window :
            The shingling window size.
        
        Return
        ------
        The shingling list.
        """
        shingling = []
        for i in range(len(text) - window + 1):
            shingling.append(text[i:i + window])
        return list(set(shingling))

    def onehot(self, doc: 'str'):
        """
        Convert a doc to a one-hot vector.

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

    def create_signature_idx(self, digit: 'int' = 20) -> 'list(list(int))':
        """Create digit-many lists. Those lists are shuffled integers from [0, ..., |V|-1]
        This function is a preparation step for the MinHash.

        Parameters
        ----------
        digit :
            The digit (length) of the MinHash signature.
        
        Return
        ------
        A list of shuffled lists of signature indices.
        """
        sig_idx_list = []
        for _ in range(digit):
            sig_idx = list(range(self.vocab_size))
            random.shuffle(sig_idx)
            sig_idx_list.append(sig_idx)
        return sig_idx_list

    def minhash(self, onehot, sig_idx_list: 'list(list(int))'):
        """For each number in [0, ..., |V|-1], find it from a sub-list of sig_idx_list, and get its 'index', 
        then look back to the one-hot vector. If the one-hot vector at that 'index' is 1, keep this 'index' 
        as a signature digit and continue. As for the next step of this iteration, use the next sub-list of 
        sig_idx_list.
        """
        signature = []
        for sub_list in sig_idx_list:
            for num in range(self.vocab_size):
                idx = sub_list.index(num)
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

        Return
        ------
        The cutted signature vector.
        """
        assert len(signature
                   ) % k == 0, 'Length of signature should be divisible by b!'
        banding = []
        window = int(len(signature) / k)
        for i in range(0, len(signature), window):
            banding.append(signature[i:i + window])
        return banding

    # def jaccard(self, a, b):
    #     a = set(a)
    #     b = set(b)
    #     return len(a.intersection(b)) / len(a.union(b))

    def compare(self, *bandings):
        for b1, b2 in zip(*bandings):
            if b1 == b2:
                return True
        else:
            return False


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

    hash_idx_list = model.create_signature_idx(20)
    print('Hash indices: ', hash_idx_list)
    print('Hash indices size: ', model.vocab_size * len(hash_idx_list))

    sig1 = model.minhash(onehot1, hash_idx_list)
    sig2 = model.minhash(onehot2, hash_idx_list)
    sig3 = model.minhash(onehot3, hash_idx_list)
    print('Signature1: ', sig1)
    print('Signature1 size: ', len(sig1))
    print('Signature2: ', sig2)
    print('Signature2 size: ', len(sig2))
    print('Signature3: ', sig3)
    print('Signature3 size: ', len(sig3))

    banding1 = model.banding(sig1, 5)
    banding2 = model.banding(sig2, 5)
    banding3 = model.banding(sig3, 5)
    print('Banding1 :', banding1)
    print('Banding2 :', banding2)
    print('Banding3 :', banding3)

    print(model.compare(banding1, banding2))
    print(model.compare(banding2, banding3))