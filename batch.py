def create_batches(dataset, max_batch_size, max_sent_size):
    """
    Create batches with sentences of similar lengths to make training more efficient.
    :param dataset: dataset
    :param max_batch_size: int
    :return: batches [(start, length), ...]
    """
    dataset.sort(key=lambda t: len(t[0]), reverse=True)  # sort by len of sent (longer first)
    source = [x[0] for x in dataset]
    src_lengths = [len(x) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(len(src_lengths)):  # skip sentences longer than max_sent_size and then start creating batches
        if src_lengths[i] > max_sent_size:
            prev = src_lengths[i + 1]
            prev_start = i + 1
    for i in range(prev_start + 1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:  # start a new batch
            batches.append((prev_start, batch_size))
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:  # continue the batch
            batch_size += 1
    return batches


def prepare_masks(sents, eos_symbol):
    """
    Mask tgt padding at end of sentence (necessary so all tgt sentences can be treated as having the same length)
    :param sents:
    :param eos_symbol:
    :return:
    """
    sents_by_words = []
    masks = []
    num_words = 0
    for i in range(len(max(sents, key=len))):
        sents_by_words.append([sent[i] if len(sent) > i else eos_symbol for sent in sents])
        mask = [(1 if len(sent) > i else 0) for sent in sents]
        masks.append(mask)
        num_words += sum(mask)
    return sents_by_words, masks, num_words
