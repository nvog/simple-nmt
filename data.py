def read_bitext(src_vocab, tgt_vocab, src_fname, tgt_fname):
    """
    Read a bitext with parallel source and target sentences.
    :param src_fname: Source bitext filename
    :param tgt_fname: Target bitext filename
    :return: yields a parallel sentence pair
    """
    with open(src_fname, "r", encoding='utf-8') as src_f, open(tgt_fname, "r", encoding='utf-8') as tgt_f:
        for line_src, line_tgt in zip(src_f, tgt_f):
            sent_src = [src_vocab[x] for x in line_src.strip().split() + ['</s>']]
            sent_tgt = [tgt_vocab[x] for x in ['<s>'] + line_tgt.strip().split() + ['</s>']]
            yield (sent_src, sent_tgt)
