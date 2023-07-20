import math

"""
This implementation is done by following Wikipedia's Viterbi Algorithm
https://en.wikipedia.org/wiki/Viterbi_algorithm
"""

class Viterbi():

    def __init__(self, O, S, Y, A, B):

        self.Obs_seq = O  # sequence of observed space (vocabulary)
        self.State_seq = S  # sequence of states (tag set)
        self.Given_seq = Y  # Sequence of observations (given sentences)
        self.A = A  # Transition matrix
        self.B = B  # Emission matrix

        self.len_states = len(self.State_seq)

        # Lookup Table for indexing words
        self.lookup = {}
        for i, word in enumerate(self.Obs_seq):
            self.lookup[word] = i

        # T1 stores the probability of the most likely path
        # T2 stores the  the most likely path
        self.len_sentence = len(Y)
        self.T1 = [[0] * self.len_sentence for i in range(self.len_states)]
        self.T2 = [[0] * self.len_sentence for i in range(self.len_states)]

        # Predicted tags
        self.Pred_Tags = [0] * self.len_sentence

    """
    This function returns the optimal sequence of tags Pred_Tags
    """
    def decode(self):

        # Initializes the start probabilities
        self.init()
        # Forward step
        self.forward()
        # Backward step
        self.backward()
        return self.Pred_Tags

    """
    Initializes the start probabilities
    """
    def init(self):

        s_idx = self.State_seq.index("<s>")         # index of fake start tag <s>
        for i in range(self.len_states):
            if self.A[s_idx][i] == 0:
                self.T1[i][0] = float("-inf")       # probability for <s> is defined as negative inifinity
                self.T2[i][0] = 0
            else:
                self.T1[i][0] = math.log(self.A[s_idx][i]) + math.log(self.B[i][self.lookup[self.Given_seq[0]]])
                self.T2[i][0] = 0


    def forward(self):

        for i in range(1, self.len_sentence):

            # if i % 5000 == 0:
            #     print("Successfully Processed {0} Words".format(i))

            for j in range(self.len_states):

                optimal_prob = float("-inf")
                optimal_path = [0] * self.len_states

                for k in range(self.len_states):

                    prob = self.T1[k][i - 1] + math.log(self.A[k][j]) + math.log(self.B[j][self.lookup[self.Given_seq[i]]])

                    if prob > optimal_prob:
                        optimal_prob = prob
                        optimal_path = k

                self.T1[j][i] = optimal_prob
                self.T2[j][i] = optimal_path


    def backward(self):

        z = [0] * self.len_sentence
        argmax = self.T1[0][self.len_sentence - 1]

        for k in range(1, self.len_states):
            if self.T1[k][self.len_sentence - 1] > argmax:
                argmax = self.T1[k][self.len_sentence - 1]
                z[self.len_sentence - 1] = k

        self.Pred_Tags[self.len_sentence - 1] = self.State_seq[z[self.len_sentence - 1]]

        for i in range(self.len_sentence - 1, 0, -1):
            z[i - 1] = self.T2[z[i]][i]
            self.Pred_Tags[i - 1] = self.State_seq[z[i - 1]]
