import collections


def get_stats(vocab):
    """
    统计词汇表中所有相邻符号对的频率。
    参数:
      vocab: dict, 键为空格分隔的符号序列（字符串），值为该词的出现次数。
    返回:
      pairs: dict, 键为相邻符号对（tuple），值为出现频率。
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        # 遍历相邻符号对
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """
    将词汇表中所有出现指定符号对的地方合并为一个新符号。
    参数:
      pair: tuple, 要合并的符号对 (a, b)。
      vocab: dict, 当前的词汇表。
    返回:
      out_vocab: dict, 更新后的词汇表。
    """
    out_vocab = {}
    # 将待合并的符号对表示为字符串形式（以空格连接）
    " ".join(pair)
    for word, freq in vocab.items():
        symbols = word.split()
        new_symbols = []
        i = 0
        while i < len(symbols):
            # 如果当前和下一个符号构成待合并的 pair，则合并
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(pair[0] + pair[1])
                i += 2  # 跳过下一个符号
            else:
                new_symbols.append(symbols[i])
                i += 1
        # 将更新后的符号序列重新组合成字符串作为新词
        new_word = " ".join(new_symbols)
        out_vocab[new_word] = freq
    return out_vocab


# 示例语料库：每个词已经用空格分割，并在词尾添加了结束符 </w>
vocab = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}

num_merges = 10  # 定义希望执行的合并次数

print("初始词汇表：")
for word, freq in vocab.items():
    print(f"{word}  {freq}")
print("=" * 30)

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # 选择出现频率最高的符号对进行合并
    best = max(pairs, key=pairs.get)
    print(f"第 {i+1} 次合并: {best} ，频率为 {pairs[best]}")
    vocab = merge_vocab(best, vocab)
    print("更新后的词汇表：")
    for word, freq in vocab.items():
        print(f"{word}  {freq}")
    print("-" * 30)
