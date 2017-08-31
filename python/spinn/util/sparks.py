# coding: utf-8

"""
Sparklines in ascii
source: https://github.com/rory/ascii_sparks/blob/master/ascii_sparks.py

"""

parts = u' ▁▂▃▄▅▆▇▉'
hex_str = '0123456789ABCDEFX'
dec_str = '0123456789X'


def sparks(nums, parts=parts):
    fraction = 1. / float(len(parts) - 1)
    return ''.join(parts[int(round(x / fraction))]
                   for x in nums).encode('utf-8')
