import os
import json
import numpy as np
import cPickle
import itertools
from collections import defaultdict
from os.path import join as pjoin


def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]


def load_ptb(data_dir, create_pkl=True):
    '''
    Load a character-based version of the text8 corpus.
    '''

    if create_pkl:
        tr_text = open(os.path.join(data_dir, 'ptb.train.txt')).readlines()
        va_text = open(os.path.join(data_dir, 'ptb.valid.txt')).readlines()
        te_text = open(os.path.join(data_dir, 'ptb.test.txt')).readlines()
        # initialize arrays for holding data
        tr_toks, va_toks, te_toks = [], [], []

        # initialize maps for fetching chars <-> ints
        word2id = {'__pad__': 0, '</S>': 1}
        id2word = ['__pad__', '</S>']

        for text, toks in ((tr_text, tr_toks),
                           (va_text, va_toks),
                           (te_text, te_toks)):
            for line in text:
                line = line.split() + ['</S>']
                for word in line:
                    if word not in word2id:
                        word2id[word] = len(word2id)
                        id2word.append(word)
                    toks.append(word2id[word])

        # dict for easy handling
        data_dict = {'train': np.asarray(tr_toks),
                     'valid': np.asarray(va_toks),
                     'test': np.asarray(te_toks),
                     'word2idx': word2id,
                     'idx2word': id2word}

        # dump the processed data for easier reloading
        f_handle = file(os.path.join(data_dir, 'ptb_dataset.pkl'), 'wb')
        cPickle.dump(data_dict, f_handle, protocol=-1)
        f_handle.close()
    else:
        # load preprocessed data
        data_dict = cPickle.load(
            open(os.path.join(data_dir, 'ptb_dataset.pkl')))
    return data_dict


def load_enwik8(data_dir, create_pkl=True):
    '''
    Load the enwik8 dataset.
    '''
    if create_pkl:
        # get paths to raw text files
        filename = os.path.join(data_dir, 'enwik8.txt')
        text = open(filename, 'r').read()

        nchars = 5000000
        tr_text = text[:-2 * nchars]
        va_text = text[-2 * nchars:-1 * nchars]
        te_text = text[-1 * nchars:]

        # initialize arrays for holding data
        tr_chars, va_chars, te_chars = [], [], []

        # initialize maps for fetching chars <-> ints
        word2id = {'__pad__': 0, '__go__': 1}
        id2word = ['__pad__', '__go__']

        # scan text files and convert to lists of int-valued keys
        for f_text, f_chars in [(tr_text, tr_chars),
                                (va_text, va_chars),
                                (te_text, te_chars)]:
            for c in f_text:
                if not (c in word2id):
                    word2id[c] = len(id2word)
                    id2word.append(c)
                f_chars.append(word2id[c])

        # dict for easy handling
        data_dict = {'train': np.asarray(tr_chars),
                     'valid': np.asarray(va_chars),
                     'test': np.asarray(te_chars),
                     'word2idx': word2id,
                     'idx2word': id2word}

        # dump the processed data for easier reloading
        f_handle = file(os.path.join(data_dir, 'enwik8_dataset.pkl'), 'wb')
        cPickle.dump(data_dict, f_handle, protocol=-1)
        f_handle.close()
    else:
        # load preprocessed data
        data_dict = cPickle.load(
            open(os.path.join(data_dir, 'enwik8_dataset.pkl')))
    return data_dict


def load_text8(data_dir, level='char', create_pkl=True):
    '''
    Load the text8 dataset.
    '''
    if create_pkl:
        # get paths to raw text files
        tr_file = os.path.join(data_dir, 'text8.train.txt')
        te_file = os.path.join(data_dir, 'text8.test.txt')

        tr_text = open(tr_file, 'r').read()
        te_text = open(te_file, 'r').read()

        word2id = {'__pad__': 0, '__go__': 1}
        id2word = ['__pad__', '__go__']
        word2tf = {}

        tr_toks, te_toks = ([], [])

        # scan text files and convert to lists of int-valued keys
        for f_text, f_toks in [(tr_text, tr_toks),
                               (te_text, te_toks)]:
            if (level == 'word'):
                f_text = f_text.split()
            for tok in f_text:
                if not (tok in word2tf):
                    word2tf[tok] = 0
                word2tf[tok] += 1

        # sort word ids by frequency
        mc_wrd = sorted(word2tf.items(), key=lambda x: x[1], reverse=True)
        for wrd, _ in mc_wrd:
            word2id[wrd] = len(id2word)
            id2word.append(wrd)

        for f_text, f_toks in [(tr_text, tr_toks),
                               (te_text, te_toks)]:
            if (level == 'word'):
                f_text = f_text.split()
            for tok in f_text:
                assert (tok in word2id), 'something went wrong'
                f_toks.append(word2id[tok])

        # build valid dataset ~ 1% of training tokens
        tr_len = int(len(tr_toks) * 0.99)
        va_toks = tr_toks[tr_len:]
        tr_toks = tr_toks[:tr_len]

        print('-- created t8 dataset, vocabulary: {}, tr_toks: {}, va_toks: {}, te_toks: {}'.format(
              len(word2id), len(tr_toks), len(va_toks), len(te_toks)))

        # dict for easy handling
        data_dict = {'train': np.asarray(tr_toks),
                     'valid': np.asarray(va_toks),
                     'test': np.asarray(te_toks),
                     'word2idx': word2id,
                     'idx2word': id2word}

        # dump the processed data for easier reloading
        with open(os.path.join(data_dir, 't8_dataset_{}.pkl'.format(level)), 'wb') as f:
            cPickle.dump(data_dict, f, protocol=-1)

    else:
        # load preprocessed data
        with open(os.path.join(data_dir, 't8_dataset_{}.pkl'.format(level)), 'rb') as f:
            data_dict = cPickle.load(f)

    return data_dict


def load_imdb_jmars(dir_path=None, max_sentence_len=16, min_sentence_len=5, topk=None):
    ''' Loads the IMDB dataset used in JMARS [1].

    This is a collection of 350k movie reviews. Each review has the following
    attributes:
    - rating : int
        Rating, between [0, 10] given by the reviewer for this movie.
    - title : str
        Title of the movie reviewed.
    - movie : str
        ID of the movie reviewed.
    - review : str
        Text of the review.
    - link : str
        URL to the IMDB's review.
    - user : str
        ID of the user who made the review.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing dataset files.

    Returns
    -------
    data_dict : dict
        Dictionary containing the following items:
            'train': list of ndarray of int
                List of sentences composing the training set.
            'valid': list of ndarray of int
                List of sentences composing the validation set.
            'test': list of ndarray of int
                List of sentences composing the ttesting set.
            'word2idx': dict
                Mapping between words and words' IDs
            'idx2word': list
                Mapping between words' IDs and words
            'word2tf': dict
                Words frequencies
            'reviews_ids': list of int
                List of reviews' ids for each sentence in the whole dataset.
            'ratings': list of int
                List of ratings for each review in the whole dataset.

    References
    ----------
    [1] Diao, Qiming and Qiu, Minghui and Wu, Chao-Yuan and Smola,
        Alexander J. and Jiang, Jing and Wang, Chong, "Jointly Modelling Aspects,
        Ratings and Sentiments for Movie Recommendation (JMARS)" (2014),  KDD '14.
    [2] This dataset comes from http://mattmahoney.net/dc/text8.zip
    [3] Code repository https://github.com/nihalb/JMARS
    '''
    VERSION = 3
    SPECIAL_TOKENS = ['__pad__', '<S>', '</S>', '<unk>']

    if dir_path is None:
        dir_path = pjoin(".", "imdb", "data")

    if max_sentence_len is not None:
        filename = "imdb_jmars_maxlen{}_minlen{}.pkl".format(max_sentence_len, min_sentence_len)
    else:
        filename = "imdb_jmars.pkl"
        max_sentence_len = np.inf

    path = pjoin(dir_path, filename)

    # Create folder if needed.
    try:
        os.mkdir(os.path.dirname(path))
    except:
        pass

    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        data_file = os.path.join(data_dir, 'data.json')

        if not os.path.isfile(data_file):
            try:
                from urllib.request import urlretrieve
            except ImportError:
                from urllib import urlretrieve

            origin = "https://www.dropbox.com/s/0oea49j7j30y671/data.json?dl=1"
            print("Downloading data (284 Mb) from {} ...".format(origin))
            urlretrieve(origin, data_file)

        # Load the dataset and process it.
        print("Processing data ...")
        with open(data_file) as f:
            data = json.load(f)

        # Use nltk punkt to extract sentences.
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        nltk.download('punkt')  # Download resource if needed.

        n_reviews = len(data)
        ratings = np.zeros(n_reviews, dtype=np.int8)
        reviews_ids = []

        word2tf = defaultdict(lambda: 0)
        dataset = []
        for i, d in enumerate(data):
            review = d['review'].lower()

            for s in sent_tokenize(review):
                words = word_tokenize(s)
                if len(words) > max_sentence_len:
                    continue

                if len(words) < min_sentence_len:
                    continue

                for w in words:
                    word2tf[w] += 1

                dataset += [words]
                reviews_ids.append(i)

            ratings[i] = d['rating']

        # sort word ids by frequency except for reserved tokens.
        idx2word = [t for t in SPECIAL_TOKENS]
        word2idx = {w: i for i, w in enumerate(idx2word)}

        mc_wrd = sorted(word2tf.items(), key=lambda x: x[1], reverse=True)
        for wrd, _ in mc_wrd:
            word2idx[wrd] = len(idx2word)
            idx2word.append(wrd)

        # Shuffle dataset and reviews_ids. At the same time, convert words to ids.
        rng = np.random.RandomState(42)
        idx = np.arange(len(dataset))
        rng.shuffle(idx)
        dataset = [np.array([word2idx[w] for w in dataset[i]]) for i in idx]
        reviews_ids = np.array(reviews_ids)[idx]

        print("Dataset has {:,} sentences and a total of {:,} words.".format(len(dataset), sum(map(len, dataset))))

        # Split in train, valid, test sets ~ 85%, 5%, 10% of dataset tokens.
        tr_len = int(len(dataset) * 0.85)
        va_len = int(len(dataset) * 0.05)
        tr_toks = dataset[:tr_len]
        va_toks = dataset[tr_len:tr_len+va_len]
        te_toks = dataset[tr_len+va_len:]

        print('-- created imdb (JMARS) dataset, vocabulary: {:,}, tr_toks: {:,}, va_toks: {:,}, te_toks: {:,}'.format(
              len(word2idx), len(tr_toks), len(va_toks), len(te_toks)))

        # dict for easy handling
        data_dict = {'version': VERSION,
                     'train': tr_toks,
                     'valid': va_toks,
                     'test': te_toks,
                     'word2idx': word2idx,
                     'idx2word': idx2word,
                     'word2tf': dict(word2tf),
                     'reviews_ids': reviews_ids,
                     'ratings': ratings}

        # dump the processed data for easier reloading
        with open(path, 'wb') as f_handle:
            cPickle.dump(data_dict, f_handle, protocol=-1)

    else:
        # load preprocessed data
        with open(path, 'rb') as f_handle:
            data_dict = cPickle.load(f_handle)

    assert data_dict.get('version', 1) >= VERSION, "Old version dectected. Delete dataset and reprocess it."

    if topk is not None:
        topk += len(SPECIAL_TOKENS)  # Take in account __pad__ and <unk>
        # -= Keep the most K frequent words =-
        unk_id = data_dict['word2idx']["<unk>"]

        def _prune(dataset):
            for i, sentence in enumerate(dataset):
                for j, e in enumerate(sentence):
                    if e >= topk:
                        dataset[i][j] = unk_id

        _prune(data_dict['train'])
        _prune(data_dict['valid'])
        _prune(data_dict['test'])

        data_dict['word2idx'] = {w: i for w, i in data_dict['word2idx'].items() if i < topk}
        data_dict['idx2word'] = data_dict['idx2word'][:topk]

        # Check integrity
        assert len(data_dict['idx2word']) == topk
        assert len(data_dict['word2idx']) == topk
        assert np.all([np.all(s < topk) for s in data_dict['train']])
        assert np.all([np.all(s < topk) for s in data_dict['valid']])
        assert np.all([np.all(s < topk) for s in data_dict['test']])

        print("Keeping top {:,} most frequent words.".format(topk-2))

    data_dict["name"] = "IMDB"
    data_dict["level"] = "w"
    data_dict["max_seq_len"] = max_sentence_len
    data_dict["min_seq_len"] = min_sentence_len
    data_dict["vocab_size"] = len(data_dict["idx2word"])
    data_dict["vocab_path"] = path[:-4] + ".tsv"

    if not os.path.isfile(data_dict["vocab_path"]):
        with open(data_dict["vocab_path"], "w") as f:
            f.write("\n".join(data_dict['idx2word']).encode("utf8") + "\n")

    return data_dict


class Text8():
    def __init__(self, data_path, seq_len, batch_size, level="word", rng_seed=1234):
        # load ptb word sequence
        text8_data = load_text8(data_path, level=level, create_pkl=False)

        self.tr_words = text8_data['train']
        self.va_words = text8_data['valid']
        self.te_words = text8_data['test']
        self.word2idx = text8_data['word2idx']
        self.idx2word = text8_data['idx2word']

        print('idx2word:')
        print(", ".join(["{}".format(v) for v in self.idx2word[:50]]))
        print('tr_words.shape: {}'.format(self.tr_words.shape))
        print('va_words.shape: {}'.format(self.va_words.shape))
        print('te_words.shape: {}'.format(self.te_words.shape))
        print('voc_size: {}'.format(len(self.idx2word)))

        self.batch_size = batch_size
        self.voc_size = len(self.idx2word)       # # of possible words
        self.seq_len = seq_len                   # length of input sequences
        self.pad_id = self.word2idx['__pad__']
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)

    def _sample_subseqs(self, source_seq, seq_count, seq_len):
        '''
        Sample subsequences of the given source sequence.
        '''
        source_len = source_seq.shape[0]
        max_start_idx = source_len - seq_len
        # sample the "base" sequences
        start_idx = self.rng.randint(low=0, high=max_start_idx, size=(seq_count,))
        idx_seqs = []
        for i in range(seq_count):
            subseq = source_seq[start_idx[i]:(start_idx[i] + seq_len)]
            idx_seqs.append(subseq[np.newaxis, :])
        idx_seqs = np.vstack(idx_seqs)
        return idx_seqs.astype("int64")

    def _prepare_heldout_set(self, source_seq):
        '''build validation/test set
        '''
        seg_size = (source_seq.shape[0] + self.batch_size - 1) / self.batch_size
        source_seq = source_seq.reshape((1, source_seq.shape[0]))
        # split into nbatch lines
        x = [source_seq[:, i:i + seg_size] for i in range(0, source_seq.shape[1], seg_size)]
        # handle padding of the last segment
        x_last = np.zeros((1, seg_size), dtype='int64') + self.pad_id
        x_last[:, :x[-1].shape[1]] = x[-1]
        x[-1] = x_last
        # concatenate in nbatch lines
        x = np.concatenate(x, axis=0)
        # start with 0 vector (first token prediction)
        x = np.concatenate([np.zeros((self.batch_size, 1)), x], axis=1)
        x = x.astype("int64")
        Xva = x[:, :-1]
        Yva = x[:, 1:]
        # split into batches
        X = [Xva[:, i:i + self.seq_len] for i in range(0, Xva.shape[1], self.seq_len)]
        Y = [Yva[:, i:i + self.seq_len] for i in range(0, Yva.shape[1], self.seq_len)]
        M = [np.not_equal(y, 0).astype('float32') for y in Y]
        return X, Y, M

    def _prepare_training_set(self, source_seq, n_batches=500):
        '''
        Prepare a sample from the source sequence.
        '''
        # sample a batch of one-hot subsequences from the source sequence
        x = self._sample_subseqs(source_seq, self.batch_size, self.seq_len * n_batches)
        x = np.concatenate([np.zeros((self.batch_size, 1)), x], axis=1)
        x = x.astype("int64")
        x_in = x[:, :-1]
        y_in = x[:, 1:]
        X = [x_in[:, i:i + self.seq_len] for i in range(0, x_in.shape[1], self.seq_len)]
        Y = [y_in[:, i:i + self.seq_len] for i in range(0, y_in.shape[1], self.seq_len)]
        M = [np.not_equal(y, 0).astype('float32') for y in Y]
        assert (len(X) == n_batches)
        assert (len(Y) == n_batches)
        assert (len(M) == n_batches)
        return X, Y, M

    def get_train_batch(self):
        X, Y, M = self._prepare_training_set(self.tr_words)
        for batch in zip(X, Y, M):
            yield batch

    def get_valid_batch(self):
        X, Y, M = self._prepare_heldout_set(self.va_words)
        for batch in zip(X, Y, M):
            yield batch

    def get_test_batch(self):
        X, Y, M = self._prepare_heldout_set(self.te_words)
        for batch in zip(X, Y, M):
            yield batch


class IMDB_JMARS():
    def __init__(self, data_path, seq_len, batch_size,
                 topk=16000, rng_seed=1234):
        # load ptb word sequence
        data = load_imdb_jmars(
            data_path, max_sentence_len=seq_len,
            min_sentence_len=5, topk=topk)

        self.tr_words = data['train']
        self.va_words = data['valid']
        self.te_words = data['test']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']

        print('idx2word:')
        print(", ".join(["{}".format(v) for v in self.idx2word[:50]]))
        print('tr_words.shape: {}'.format(len(self.tr_words)))
        print('va_words.shape: {}'.format(len(self.va_words)))
        print('te_words.shape: {}'.format(len(self.te_words)))
        print('voc_size: {}'.format(len(self.idx2word)))

        self.batch_size = batch_size
        self.voc_size = len(self.idx2word)      # # of possible words
        self.seq_len = seq_len                  # length of input sequences
        self.pad_id = self.word2idx['__pad__']  # pad token
        self.bos_id = self.word2idx['<S>']      # beginning of sentence
        self.eos_id = self.word2idx['</S>']     # end of sentence
        self.unk_id = self.word2idx['<unk>']    # unk token
        assert self.pad_id == 0
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)

        # Indices for trainset
        self.indices_trainset = np.arange(len(self.tr_words))

    def pad_batch(self, sentences):
        """
        Pad each sentence with __pad__ token so they all have the same length.
        """
        max_len = max(map(len, sentences))
        batch = self.pad_id * np.ones((len(sentences), max_len), dtype=int)
        for i, sentence in enumerate(sentences):
            batch[i, :len(sentence)] = sentence
        return batch

    def pad_sent(self, sentence, max_len=20):
        """
        Pad each sentence with __pad__ token so they all have the same length.
        """
        batch = self.pad_id * np.ones((1, max_len), dtype=int)
        batch[0, :len(sentence)] = sentence
        return batch

    def prepare_batch(self, sentences):
        """
        Add <S> and </S> tokens to each sentence and pad the batch.
        """
        def _add_special_symbols(s):
            return [self.bos_id] + list(s) + [self.eos_id]

        sents = [_add_special_symbols(s) for s in sentences]
        batch = self.pad_batch(sents)
        x = batch[:, :-1]
        y = batch[:, 1:]
        m = np.not_equal(y, self.pad_id).astype('float32')
        return x, y, m

    def batch2text(self, batch, eos_id=None):
        sentences = []
        for i, s in enumerate(batch):
            sentence = []
            for idx in s:
                sentence.append(self.idx2word[idx])
                if eos_id == idx:
                    break
            sentences.append(" ".join(sentence))
        return sentences

    def print_batch(self, batch, eos_id=None, print_number=True):
        for i, s in enumerate(batch):
            sentence = []
            for idx in s:
                sentence.append(self.idx2word[idx])
                if eos_id == idx:
                    break

            if print_number:
                print("{}. ".format(i) + " ".join(sentence))
            else:
                print(" ".join(sentence))

    def get_train_batch(self, shuffle=True):
        if shuffle:
            self.rng.shuffle(self.indices_trainset)  # In-place

        for indices in chunk(self.indices_trainset, n=self.batch_size):
            if len(indices) != self.batch_size:
                continue  # Feeling lazy, skip incomplete batch.

            batch = [self.tr_words[idx] for idx in indices]
            yield self.prepare_batch(batch)

    def get_valid_batch(self):
        for batch in chunk(self.va_words, n=self.batch_size):
            yield self.prepare_batch(batch)

    def get_test_batch(self):
        for batch in chunk(self.te_words, n=self.batch_size):
            yield self.prepare_batch(batch)


class PTB():
    def __init__(self, data_path, seq_len, batch_size, rng_seed=1234):
        # load ptb word sequence
        text8_data = load_ptb(data_path, create_pkl=True)

        self.tr_words = text8_data['train']
        self.va_words = text8_data['valid']
        self.te_words = text8_data['test']
        self.word2idx = text8_data['word2idx']
        self.idx2word = text8_data['idx2word']

        print('idx2word:')
        print(", ".join(["{}".format(v) for v in self.idx2word[:50]]))
        print('tr_words.shape: {}'.format(self.tr_words.shape))
        print('va_words.shape: {}'.format(self.va_words.shape))
        print('te_words.shape: {}'.format(self.te_words.shape))
        print('voc_size: {}'.format(len(self.idx2word)))

        self.pad_id = self.word2idx['__pad__']
        self.unk_id = self.word2idx['<unk>']
        self.bos_id = self.word2idx['</S>']
        self.eos_id = self.word2idx['</S>']
        self.batch_size = batch_size
        self.voc_size = len(self.idx2word)  # # of possible words
        self.seq_len = seq_len              # length of input sequences
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)

    def _sample_subseqs(self, source_seq, seq_count, seq_len):
        '''
        Sample subsequences of the given source sequence.
        '''
        source_len = source_seq.shape[0]
        max_start_idx = source_len - seq_len
        # sample the "base" sequences
        start_idx = self.rng.randint(
            low=0, high=max_start_idx, size=(seq_count,))
        idx_seqs = []
        for i in range(seq_count):
            subseq = source_seq[start_idx[i]:(start_idx[i] + seq_len)]
            idx_seqs.append(subseq[np.newaxis, :])
        idx_seqs = np.vstack(idx_seqs)
        return idx_seqs.astype("int64")

    def _prepare_heldout_set(self, source_seq):
        '''build validation/test set
        '''
        seg_size = (source_seq.shape[0] + self.batch_size - 1) / self.batch_size
        source_seq = source_seq.reshape((1, source_seq.shape[0]))
        # split into nbatch lines
        x = [source_seq[:, i:i + seg_size] for i in range(0, source_seq.shape[1], seg_size)]
        # handle padding of the last segment
        x_last = np.zeros((1, seg_size), dtype='int64') + self.pad_id
        x_last[:, :x[-1].shape[1]] = x[-1]
        x[-1] = x_last
        # concatenate in nbatch lines
        x = np.concatenate(x, axis=0)
        # start with 0 vector (first token prediction)
        x = np.concatenate([np.zeros((self.batch_size, 1)) + self.bos_id, x], axis=1)
        x = x.astype("int64")
        Xva = x[:, :-1]
        Yva = x[:, 1:]
        # split into batches
        X = [Xva[:, i:i + self.seq_len] for i in range(0, Xva.shape[1], self.seq_len)]
        Y = [Yva[:, i:i + self.seq_len] for i in range(0, Yva.shape[1], self.seq_len)]
        M = [np.not_equal(y, 0).astype('float32') for y in Y]
        return X, Y, M

    def _prepare_training_set(self, source_seq, n_batches=500):
        '''
        Prepare a sample from the source sequence.
        '''
        # sample a batch of one-hot subsequences from the source sequence
        x = self._sample_subseqs(source_seq, self.batch_size, self.seq_len * n_batches)
        x = np.concatenate([np.zeros((self.batch_size, 1)) + self.bos_id, x], axis=1)
        x = x.astype("int64")
        x_in = x[:, :-1]
        y_in = x[:, 1:]
        X = [x_in[:, i:i + self.seq_len] for i in range(0, x_in.shape[1], self.seq_len)]
        Y = [y_in[:, i:i + self.seq_len] for i in range(0, y_in.shape[1], self.seq_len)]
        M = [np.not_equal(y, 0).astype('float32') for y in Y]
        assert (len(X) == n_batches)
        assert (len(Y) == n_batches)
        assert (len(M) == n_batches)
        return X, Y, M

    def get_train_batch(self):
        X, Y, M = self._prepare_training_set(self.tr_words)
        for batch in zip(X, Y, M):
            yield batch

    def get_valid_batch(self):
        X, Y, M = self._prepare_heldout_set(self.va_words)
        for batch in zip(X, Y, M):
            yield batch

    def get_test_batch(self):
        X, Y, M = self._prepare_heldout_set(self.te_words)
        for batch in zip(X, Y, M):
            yield batch
