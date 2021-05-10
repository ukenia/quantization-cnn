from Levenshtein import distance as levenshtein_distance

LETTER_LIST = ['<', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '>']


def calc_levenshtein_dist(list1, list2):
    dist = 0
    for index in range(len(list1)):
        dist += levenshtein_distance(list1[index], list2[index])

    return dist / len(list1)


def unpad(padded_seq, lengths):
    unpadded_list = []
    for index in range(padded_seq.shape[0]):
        unpadded_list.append(padded_seq[index][0][:lengths[index][0]])

    return unpadded_list


def unpad_beam1(padded_seq, lengths):
    unpadded_list = []
    for index in range(len(lengths)):
        unpadded_list.append(padded_seq[index, :lengths[index]])

    return unpadded_list


def generate_output(class_labels):
    out_string = ""
    for label in class_labels:
        if label == 33:
            break
        if label == 0:
            continue
        else:
            out_string += LETTER_LIST[label]
    return out_string


def generate_all_outputs(labels_all):
    output_strings = []
    for batch in labels_all:
        for class_labels in batch:
            output_strings.append(generate_output(class_labels))

    return output_strings