from spinn import util

NUMBERS = list(range(-10, 11))

FIXED_VOCABULARY = {str(x): i + 1 for i, x in enumerate(NUMBERS)}
FIXED_VOCABULARY.update({
    util.PADDING_TOKEN: 0,
    "+": len(FIXED_VOCABULARY) + 1,
    "-": len(FIXED_VOCABULARY) + 2
})
assert len(set(FIXED_VOCABULARY.values())) == len(list(FIXED_VOCABULARY.values()))
