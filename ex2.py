import re
import sys
import random
import math
from collections import defaultdict, Counter
import nltk


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should support the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.error_tables = self.init_error_tables()
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        self.denominator_indices_map = {'insertion': {'start': 0, 'end': 1},
                                        'deletion': {'start': 0, 'end': 2},
                                        'substitution': {'start': 0, 'end': 1},
                                        'transposition': {'start': 0, 'end': 2}}
        self.letters_count = {}
        if lm is not None:
            self.words_count = self.count_unigrams()

    def count_unigrams(self):
        """
        counts the unigrams in the given language model
        :return: a dictionary of {word: count}
        """
        ngram_model = self.lm.get_model()
        words = []
        for ngram in ngram_model:
            ngram_split = ngram.split()
            words.append(ngram_split[0])
        return Counter(words)

    @staticmethod
    def init_error_tables():
        """
        creates an initialized error table with four dicts (insertion, deletion, substitution, transposition)
        with {letters: count}
        :return:
        """
        error_tables = {'insertion': dict(), 'deletion': dict(), 'substitution': dict(), 'transposition': dict()}
        letters = 'abcdefghijklmnopqrstuvwxyz# ,.?!\'()'
        for letter1 in letters:
            for letter2 in letters:
                error_tables['insertion'][letter1+letter2] = 0
                error_tables['deletion'][letter1+letter2] = 0
                error_tables['substitution'][letter1+letter2] = 0
                error_tables['transposition'][letter1+letter2] = 0
        return error_tables

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """
        self.lm = Ngram_Language_Model(n, False)
        self.lm.build_model(normalize_text(text))
        self.words_count = self.count_unigrams()
        self.letters_count = {}
        return self.lm

    def add_language_model(self, lm):
        """
        Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a language model object
        """
        if lm is not None:
            self.lm = lm
            self.words_count = self.count_unigrams()
            self.letters_count = {}

    def learn_error_tables(self, errors_file):
        """
        Returns a nested dictionary {str:dict} where str is in:
        <'deletion', 'insertion', 'transposition', 'substitution'> and the
        inner dict {str: int} represents the confusion matrix of the
        specific errors, where str is a string of two characters matching the
        row and column "indices" in the relevant confusion matrix and the int is the
        observed count of such an error (computed from the specified errors file).
        Examples of such string are 'xy', for deletion of a 'y'
        after an 'x', insertion of a 'y' after an 'x'  and substitution
        of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.
            Notes:
                1. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>
            Returns:
                A dictionary of confusion "matrices" by error type (dict).
        """
        with open(errors_file, 'r') as err_file:
            for line in err_file:
                line_split = line.strip().split('\t')
                if len(line_split) != 2:
                    continue
                wrong_word = normalize_text(line_split[0])
                correct_word = normalize_text(line_split[1])

                edits1 = self.edits(wrong_word)  # find edits with one step
                self.add_errors_to_table(edits1, correct_word)

                # new_candidates = [value[2] for value in edits1]
                # for new_wrong_word in new_candidates:
                #     edits2 = self.edits(correct_word)  # find edits with 2 steps
                #     self.add_errors_to_table(edits2, new_wrong_word)

    def edits(self, word):
        word = '#' + word
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        inserts = [('insertion', L[-1] + c, L[1:] + c + R) for L, R in splits[1:] for c in self.letters]
        deletes = [('deletion', L[-1] + R[0], L[1:] + R[1:]) for L, R in splits[1:] if R]
        transposes = [('transposition', R[1] + R[0], L[1:] + R[1] + R[0] + R[2:]) for L, R in splits[1:] if len(R) > 1]
        replaces = [('substitution', R[0] + c, L[1:] + c + R[1:]) for L, R in splits[1:] if R for c in self.letters]
        edits = inserts + deletes + transposes + replaces
        return edits

    def add_errors_to_table(self, edits, word):
        for error_type, letters, optional_word in edits:
            if optional_word == word:
                self.error_tables[error_type][letters] += 1

    def add_error_tables(self, error_tables):
        """
        Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        if error_tables is not None:
            self.error_tables = error_tables

    def evaluate(self, text):
        """
        Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        if self.lm is not None:
            return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """
        Returns the most probable fix for the specified text. Use a simple
        noisy channel model if the number of tokens in the specified text is
        smaller than the length (n) of the language model.
            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        text = normalize_text(text)
        sentence_split = text.split()

        sentences_eval = {}  # save sentences evaluation
        final_candidates = []  # save final candidates for word correction
        curr_word_candidate_keep, curr_word_candidate_keep_prob = [], []  # save word-wise candidates
        unknown_words = [self.known(word) for word in sentence_split]  # check if words in sentence are known
        all_known = all(unknown_words)  # whether all words are correct

        for word_idx, word in enumerate(sentence_split):
            if not all_known and unknown_words[word_idx]:  # correct all words if all are known, else correct only unknown words
                continue
            if self.known(word):  # keep the same word
                candidate_keep, evaluate_prob_keep, _ = self.get_candidate_details(word_idx, sentence_split, word, [], [('', '', word)], sentences_eval)
                curr_word_candidate_keep.append(candidate_keep)
                curr_word_candidate_keep_prob.append(evaluate_prob_keep + math.log(alpha))

            curr_word_candidates, curr_word_candidates_evaluate_probs, curr_word_candidates_probs = [], [], []
            word_edits1 = self.edits(word)  # get all possible 1 edit changes
            for edit1 in word_edits1:
                correction_candidate1 = edit1[2]  # get optional word
                if self.known(correction_candidate1):  # if is in dict
                    candidate, evaluate_prob, noisy_channel_prob = self.get_candidate_details(word_idx, sentence_split, correction_candidate1, [], [edit1], sentences_eval)
                    curr_word_candidates.append(candidate)  # add to current word candidates
                    curr_word_candidates_evaluate_probs.append(evaluate_prob)
                    curr_word_candidates_probs.append(noisy_channel_prob)  # calc probability

                word_edits2 = self.edits(correction_candidate1)  # get all possible 1 edit changes
                for edit2 in word_edits2:
                    correction_candidate2 = edit2[2]
                    if self.known(correction_candidate2):  # if is in dict
                        candidate, evaluate_prob, noisy_channel_prob = self.get_candidate_details(word_idx, sentence_split, correction_candidate2, [edit1], [edit2], sentences_eval)
                        curr_word_candidates.append(candidate)
                        curr_word_candidates_evaluate_probs.append(evaluate_prob)
                        curr_word_candidates_probs.append(noisy_channel_prob)  # calc probability

            curr_word_candidates_probs = self.normalize_noisy_channel_probs(curr_word_candidates_probs, alpha)  # normalize probs to sum to 1-alpha
            final_probs = [sum(probs) for probs in zip(curr_word_candidates_evaluate_probs, curr_word_candidates_probs)]
            final_candidates.extend(zip(curr_word_candidates, final_probs))

        final_candidates.extend(zip(curr_word_candidate_keep, curr_word_candidate_keep_prob))  # add "keep" candidates to final candidates
        best_match = max(final_candidates, key=lambda x: x[1])  # get change with highest prob
        return best_match[0][0]

    def get_candidate_details(self, word_idx, sentence_split, correction_candidate, curr_edit, prev_edit, sentences_eval):
        """
        get candidate details in the form of [candidate sentence, [edit1=(correction_type, letters, suggested_word),edit2=...]]
        :param word_idx: index of the replaced word in the original sentence
        :param sentence_split: the original sentence split to words
        :param correction_candidate: the candidate word
        :param curr_edit: edit in the form of (correction_type, letters, suggested_word)
        :param prev_edit: previous edit (in case of edit distance=2)
        :param sentences_eval: sentences evaluation dict {sentence:prob}
        :return: the candidate and it's probability
        """
        optional_candidate_sentence_split = sentence_split.copy()
        optional_candidate_sentence_split[word_idx] = correction_candidate
        optional_candidate_sentence = ' '.join(optional_candidate_sentence_split)
        candidate = [optional_candidate_sentence] + prev_edit + curr_edit
        evaluate_prob, noisy_channel_prob = self.probability(candidate, sentences_eval)
        return candidate, evaluate_prob, noisy_channel_prob

    def probability(self, candidate, sentences_eval):
        """
        calculates the probability of a single candidate
        :param candidate: the candidate to compute the probability of
        :param sentences_eval: sentences evaluation dict {sentence:prob}
        :return:
        """
        sentence = candidate[0]
        if sentence in sentences_eval.keys():  # get sentence evaluation if exists
            evaluate_prob = sentences_eval[sentence]
        else:  # calc sentence evaluation
            evaluate_prob = self.lm.evaluate(sentence)
            sentences_eval[sentence] = evaluate_prob
        if candidate[1][:2] == ('', ''):  # if keeping the original word
            return evaluate_prob, None
        noisy_channel_prob = 1
        corrections = candidate[1:]
        for correction in corrections:  # calculate the noisy model probability the current correction
            corr_type = correction[0]
            two_letters = correction[1]
            nominator = self.error_tables[corr_type][two_letters] if two_letters in self.error_tables[corr_type].keys() else 0
            denom_letters = two_letters[self.denominator_indices_map[corr_type]['start']:self.denominator_indices_map[corr_type]['end']]
            denominator = self.get_count_in_corpus(denom_letters)
            if nominator == 0 or denominator == 0:  # if could not be calculated
                return evaluate_prob, -math.inf
            noisy_channel_prob *= (nominator / denominator)
        return evaluate_prob, noisy_channel_prob

    @staticmethod
    def normalize_noisy_channel_probs(curr_word_candidates_probs, alpha):
        """
        normalize the candidate probabilities of a single word change to sum to 1-alpha
        :param curr_word_candidates_probs: all the probabilities of the same word changes
        :param alpha: the alpha to be calculated by
        :return: array of normalized probabilities
        """
        probs_sum = sum([prob for prob in curr_word_candidates_probs if prob != -math.inf])
        probs_normalizer = probs_sum / (1 - alpha)
        probs_normalizer = probs_normalizer if probs_normalizer != 0 else 1
        curr_word_candidates_normalized_probs = []
        for curr_word_candidates_prob in curr_word_candidates_probs:
            if curr_word_candidates_prob != -math.inf:
                curr_word_candidates_normalized_probs.append(math.log(curr_word_candidates_prob / probs_normalizer))
            else:
                curr_word_candidates_normalized_probs.append(curr_word_candidates_prob)
        return curr_word_candidates_normalized_probs

    def known(self, word):
        """
        checks if word exists in the language model
        :param word: word to check
        :return: True if exists, False otherwise
        """
        return word in self.words_count.keys()

    def get_count_in_corpus(self, letters):
        """
        returns the count of the occurrence of specific letters
        :param letters: letters to count
        :return: count of letters
        """
        if letters in self.letters_count.keys():  # if letters are counted already
            return self.letters_count[letters]
        if letters == '#':  # sum of all letters count
            count = sum(self.words_count.values())
            self.letters_count[letters] = count
            return count
        count = 0
        for word, word_count in self.words_count.items():
            if letters[0] == '#':  # calc if letter is at the beginning of a word
                if word.startswith(letters[1]):
                    word_c = 1
                else:
                    word_c = 0
            else:
                word_c = word.count(letters)
            count += word_c * word_count
        self.letters_count[letters] = count
        return count


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a model from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """
        Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.n = n
        self.chars = chars
        self.model = defaultdict(int)

    def build_model(self, text):
        """
        populates a dictionary counting all ngrams in the specified text.
            Args:
                text (str): the text to construct the model from.
        """
        self.ngram_count = {}
        split_text = self.split_text(text)
        [self.add_ngram_to_model(self.concat_ngrams(split_text[i:i+self.n])) for i in range(len(split_text)-2)]

    def get_model(self):
        """
        Returns the model as a dictionary of the form {ngram:count}
        """
        return self.model

    def generate(self, context=None, n=20):
        """
        Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted.
            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.
            Return:
                String. The generated text.
        """
        if context is None:
            context = self.get_random_context()  # generate random context
        generated_text = self.split_text(context)

        while self.get_count(self.concat_ngrams(generated_text[-self.n + 1:])) == 0:
            generated_text.append(self.get_next_random())
        for i in range(n + 1 - len(generated_text)):
            current_context = self.concat_ngrams(generated_text[-self.n + 1:])  # extract current loop's context (last n-1 ngrams)
            next_ngram = self.get_next_by_context(current_context)  # get the next ngram
            if next_ngram is None:  # contexts are exhausted
                break
            generated_text.append(next_ngram)
        return self.concat_ngrams(generated_text)

    def evaluate(self, text):
        """
        Returns the log-likelihod of the specified text to be generated by the model.
        Laplace smoothing should be applied if necessary.
           Args:
               text (str): Text to evaluate.
           Returns:
               Float. The float should reflect the (log) probability.
        """
        split_text = self.split_text(text)
        log_likelihood = 0
        for i in range(len(split_text)-self.n+1):
            if i < self.n:  # extract relevant ngrams with smaller length than model's n.
                full_ngram = ' '.join(split_text[:i+1])
                context_ngram = ' '.join(split_text[:i])
            else:  # when reaching the n'th word of the text, evaluate on full ngram size
                full_ngram = ' '.join(split_text[i:i + self.n])
                context_ngram = ' '.join(split_text[i:i + self.n - 1])
            ngram_count = self.get_count(full_ngram)
            if ngram_count != 0:  # ngram is found in the model
                context_sum = self.get_count(context_ngram) if context_ngram != '' else sum(self.model.values())
                prob = float(ngram_count) / float(context_sum)
            else:  # ngram is not found; thus, smoothing is done
                prob = self.smooth(full_ngram)
            log_likelihood += math.log(prob)
        return log_likelihood

    def smooth(self, ngram):
        """
        Returns the smoothed (Laplace) probability of the specified ngram.
            Args:
                ngram (str): the ngram to have it's probability smoothed
            Returns:
                float. The smoothed probability.
        """
        split_text = self.split_text(ngram)
        context_ngram = self.concat_ngrams(split_text[:-1])
        N = self.get_count(context_ngram)  # context's count
        V = len(self.model)  # vocabulary size
        return 1 / float(N + V)  # laplace smoothing (1 / (N+V))

    def get_count(self, ngram_to_count):
        """
        Returns the count of a given ngram in the model
        @param ngram_to_count - the ngram to be counted
        @return: the count of a given ngram
        """
        if ngram_to_count in self.ngram_count.keys():
            return self.ngram_count[ngram_to_count]
        ngram_count = 0
        for ngram in self.model:
            ngram_split = self.split_text(ngram)
            loc = ngram_to_count.count(' ') + 1 if not self.chars else len(ngram_to_count)  # get the correct location to split ngram, based on model type
            n_gram_to_check = self.concat_ngrams(ngram_split[:loc])
            if n_gram_to_check == ngram_to_count:  # count ngram if matches
                ngram_count += self.model[ngram]
        self.ngram_count[ngram_to_count] = ngram_count
        return ngram_count

    def get_next_by_context(self, context):
        """
        Generates the next gram (word or character) based on the probabilities of the context in the LM
        (i.e., samples from the distribution of existing context probabilities)
        This function is non-deterministic due to the dynamic sampling
        @param context: the context to draw the next gram from
        @return: the next word
        """
        candidates = []
        probs = []
        context_total = self.get_count(context)
        if context_total == 0:  # context is exhausted
            return None
        for n_gram in self.model:
            if n_gram.startswith(context):  # get candidates from context
                candidate = self.split_text(n_gram)[-1]
                candidates.append(candidate)
                prob = float(self.model[n_gram]) / float(context_total)
                probs.append(prob)
        next_gram = random.choices(candidates, probs)[0]
        return next_gram

    def get_next_random(self):
        """
        Generates the next gram (word or character) based on general words distribution
        (i.e., samples from the distribution of existing general word probabilities)
        This function is non-deterministic due to the dynamic sampling
        @return: the next word
        """
        candidates = []
        probs = []
        total_count = len(self.model.values())
        for n_gram in self.model:  # get candidates and their probabilities
            candidate = self.split_text(n_gram)[-1]
            candidates.append(candidate)
            prob = float(self.model[n_gram]) / total_count
            probs.append(prob)
        next_gram = random.choices(candidates, probs)[0]
        return next_gram

    def concat_ngrams(self, ngrams_split):
        """
        Merges ngrams based on the models type (word or character)
        @param ngrams_split: the split ngrams to be merged
        @return: concatenated string
        """
        join_char = '' if self.chars else ' '
        return join_char.join(ngrams_split)

    def get_random_context(self):
        """
        Generates random context based on the distribution of ngrams probabilities in the model
        This function is non-deterministic due to dynamic sampling
        @return:
        """
        candidates = []
        probs = []
        for n_gram in self.model:
            candidate = self.concat_ngrams(self.split_text(n_gram)[:-1])  # candidate string
            candidates.append(candidate)
            prob = float(self.get_count(candidate)) / sum(self.model.values())  # candidate probability
            probs.append(prob)
        next_ngram = random.choices(candidates, probs)[0]  # get random sample from probabilities distribution
        return next_ngram

    def split_text(self, ngram):
        """
        Splits an ngram based on the model type (word or character).
        Word-level models are split by space
        Character-level models are split character-wise
        @param ngram: the ngram to split
        @return: array of ngrams elements
        """
        if self.chars:
            return [c for c in ngram]
        else:
            return ngram.split()

    def add_ngram_to_model(self, ngram):
        """
        Adds an ngram to the language model
        @param ngram: the ngram to add
        @return:
        """
        if ngram not in self.model:  # ngram doesn't exists in model
            self.model[ngram] = 1
        else:
            self.model[ngram] += 1


def normalize_text(text, remove_punctuation=False):
    """
    Returns a normalized string based on the specify string.
    This function adds space padding to punctuation - so it counts as a component in the ngram model
    Removes double spacing.
    Removing punctuation is optional.
       Args:
           text (str): the text to normalize
           remove_punctuation (str): whether punctuation should be removed from text. Default: False
       Returns:
           string. the normalized text.
    """
    text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    else:
        text = re.sub('[^a-z ,.?!()\']', '', text)  # remove all non-alphabetic+[,.?!()] words
        text = re.sub('([.,!?()])', r' \1 ', text)  # pad punctuation with space
        text = re.sub('\s{2,}', ' ', text)  # remove double spacing
    return text


def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Alon Zolfi', 'id': '205846074', 'email': 'zolfi@post.bgu.ac.il'}