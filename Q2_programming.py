from typing import Sequence

# This solution assumes the dictionary has been preprocessed into a format where
# each key is a tuple of phonemes and the corresponding value is a list of words
# that have that phoneme sequence. The dictionary is preprocessed into this format
# as follows:

# inversed_pronunciation_dict = {
#     ('AE', 'B', 'AH', 'K', 'AH', 'S'): ['ABACUS'],
#     ('B', 'UH', 'K'): ['BOOK'],
#     ('DH', 'EH', 'R'): ['THEIR', 'THERE'],
#     ('T', 'AH', 'M', 'AA', 'T', 'OW'): ['TOMATO'],
#     ('T', 'AH', 'M', 'EY', 'T', 'OW'): ['TOMATO']
# }


def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
    """
    Given a sequence of phonemes, finds all combinations of words from a preprocessed 
    pronunciation dictionary that can produce this sequence. The dictionary has been 
    preprocessed such that the keys are tuples of phonemes, and the values are lists 
    of words corresponding to those phoneme sequences.

    The function uses a recursive approach to try to match the phonemes from the beginning 
    to the end and returns all valid combinations of words that can form the given phoneme 
    sequence.

    Parameters:
    -----------
    phonemes : Sequence[str]
        A sequence of phonemes (sound units) representing a target pronunciation.

    Returns:
    --------
    Sequence[Sequence[str]]
        A list of lists, where each inner list is a sequence of words that together 
        match the given phoneme sequence. Each word corresponds to a segment of the 
        phoneme sequence.

    Example:
    --------
    phonemes = ["DH", "EH", "R", "DH", "EH", "R"]
    result = find_word_combos_with_pronunciation(phonemes)
    print(result)
    # Output: [["THEIR", "THEIR"], ["THEIR", "THERE"], ["THERE", "THEIR"], ["THERE", "THERE"]]

    """

    # If the phoneme sequence is empty, return an empty list (base case for recursion)
    if len(phonemes) < 1:
        return [[]]

    # List to store all the combinations of words
    words = []

    # Loop through each possible position in the phoneme sequence
    for i in range(len(phonemes)):
        # Slice the phoneme sequence from the start to the current position
        phonemes_slice = phonemes[:i + 1]

        # Check if the sliced phonemes match any key in the dictionary
        if tuple(phonemes_slice) in inversed_pronunciation_dict.keys():
            # If a match is found, get the corresponding words
            found_words = inversed_pronunciation_dict[tuple(phonemes_slice)]

            # For each word found, recursively process the remaining phonemes
            for word in found_words:
                # Recursively find the combinations for the remaining phonemes
                for next_word in find_word_combos_with_pronunciation(phonemes[i + 1:]):
                    # Combine the current word with the subsequent words
                    words.append([word, *next_word])

    return words
