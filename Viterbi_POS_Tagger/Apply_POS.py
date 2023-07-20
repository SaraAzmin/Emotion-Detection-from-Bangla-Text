
# tagged_data is a list a off (word, tag) tuples
# Contains newlines and <s> as start tag for a sentence
# This function will make a list of list containing each line of the original documents
# Use appropriate pos tags to reduce the number of features in the original documents

def selected_pos(tagged_data):
    '''This function only cosiders selected POS given in the list'''

    pos = ['JJ', 'CX', 'VM', 'NP', 'AMN']
    # pos = ['JJ']
    documents = []
    line = []

    for word, tag in tagged_data:
        if word == '\n':
            documents.append(line)
            line = []
            # print()
        elif tag in pos:
            line.append('{}_{}'.format(word, tag))
            # print("{0}\t{1}".format(word, tag))

    return documents


def consider_all_pos(tagged_data):
    '''This function considers all POS'''

    documents = []
    doc = []
    for word, tag in tagged_data:
        if word == '\n':
            documents.append(doc)
            doc = []
        else:
            doc.append('{}_{}'.format(word, tag))
    
    return documents


def pos_feature(tagged_data, all_pos):
    '''This function reduce feature based on selected pos'''

    if all_pos:
        return consider_all_pos(tagged_data)       # considers all POS
    else:
        return selected_pos(tagged_data)           # consisders selected POS


