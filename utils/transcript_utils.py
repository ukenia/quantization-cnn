import numpy as np

LETTER_LIST = ['<', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '>']

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    '''
    letter2index = {}
    index2letter = {}

    for index, letter in enumerate(letter_list):
        letter2index[letter] = index
        index2letter[index] = letter

    return letter2index, index2letter


def transform_letter_to_index(raw_transcripts, letter2index):
    '''
    Transforms text input to numerical input by converting each letter
    to its corresponding index from letter_list

    Args:
        raw_transcripts: Raw text transcripts with the shape of (N, )

    Return:
        transcripts: Converted index-format transcripts. This would be a list with a length of N
    '''

    index_transcripts = []
    for example in raw_transcripts:
        example_indices = []
        example_str = b' '.join(example).decode('utf-8')
        for letter in example_str:
            example_indices.append(letter2index[letter])

        # example_indices.append(len(letter2index)-1)
        index_transcripts.append(example_indices)

    return np.array(index_transcripts)


def transform_index_to_letter(transcript_indices, index2letter):
    letter_transcripts = []
    for transcript in transcript_indices:
        example_transcript = []
        for index in transcript:
            example_transcript.append(index2letter[index])
        letter_transcripts.append("".join(example_transcript))

    return letter_transcripts


# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)