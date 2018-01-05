# Not efficient, but good enough.

def span_to_example(words,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x,
        id=''):
    label = words[0][1]
    if not keep_fn(label):
        return None
    label = convert_fn(label)

    example = {}
    example["label"] = label
    example["sentence"] = " ".join(words)
    example["tokens"] = []
    example["transitions"] = []
    for index, word in enumerate(words):
        if word[0] != "(":
            if word == ")":
                # Ignore unary merges
                if words[index - 1] == ")":
                    example["transitions"].append(1)
            else:
                # Downcase all words to match GloVe.
                example["tokens"].append(word)
                example["transitions"].append(0)
    example["example_id"] = id
    return example

def convert_unary_binary_bracketed_data(
        filename,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x,
        top_node_only=False):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    if top_node_only:
        print("SST eval mode: Preserving only top node label.")
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            stack = []
            words = line.replace(')', ' )')
            words = words.split(' ')
            if top_node_only:
                example = span_to_example(words, keep_fn, convert_fn, str(len(examples)))
                if example is not None:
                    examples.append(example)
            else:
                for index, word in enumerate(words):
                    if word[0] != "(":
                        if word == ")":
                            start = stack.pop()
                            example = span_to_example(words[start:index + 1], keep_fn, convert_fn, str(len(examples)))
                            if example is not None:
                                examples.append(example)
                    else:
                        stack.append(index)
    return examples
