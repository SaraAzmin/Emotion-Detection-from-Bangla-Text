TRAIN = "Viterbi_POS_Tagger/bn_nltr/data_clean/full_corpus.txt"

# TEST = "Viterbi_POS_Tagger/bn_nltr/data_clean/test.txt"
# TEST_OUT = "Viterbi_POS_Tagger/output/test_out.txt"

MODEL = "Viterbi_POS_Tagger/model/model.txt"
VOCAB = "Viterbi_POS_Tagger/model/vocab.txt"

TAGS_UNIV = [
    "ALC",
    "JQ",
    "VAUX",
    "RDX",
    "VA",
    "RDF",
    "DRL",
    "PP",
    "NC",
    "PU",
    "CCL",
    "LV",
    "PRF",
    "LC",
    "CSD",
    "RDS",
    "CIN",
    "PPR",
    "CX",
    "DAB",
    "VM",
    "CCD",
    "JJ",
    "PRL",
    "CSB",
    "PRC",
    "DWH",
    "NP",
    "PWH",
    "NST",
    "NV",
    "AMN",
]


'''
Description of the Tagset

    "ALC" = Adv/Location
    "JQ" = Quantifier   *
    "VAUX" = Auxiliary Verb *
    "RDX" = Residuals
    "VA" = Auxiliary Verb * 
    "RDF" = Residual/Foreign Word
    "DRL" = Relative Demonstrative  *
    "PP" = Post- position
    "NC" = Common Noun
    "PU" = Punctuation
    "CCL" = Particle/Classifier
    "LV" = Participle/Verbal
    "PRF" = Pronoun/Reflexive
    "LC" = Participle/Conditional   *
    "CSD" = Participle/Subordinating
    "RDS" = Residual/Symbol
    "CIN" = Participle/Interjection
    "PPR" = Pronoun
    "CX" = Participle/Other ***
    "DAB" = Demonstrative/Absolutive
    "VM" = Main Verb    ***
    "CCD" = Participle/Co-ordinating    *
    "JJ" = Adjective *****
    "PRL" = Relative Pronoun
    "CSB" = Participle/Subordinating
    "PRC" = Pronoun/Reciprocal
    "DWH" = Wh-demostrative
    "NP" = Proper Noun  ***
    "PWH" = Wh-pronoun 
    "NST" = Spatio-temporal Noun
    "NV" = Verbal Noun * 
    "AMN" = Adv/Manner ***
'''
