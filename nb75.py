from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
#from ruleparser import RuleParser
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
import re
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

# Importing Files for POS Tagged Data
import Viterbi_POS_Tagger.HMM as HMM
import Viterbi_POS_Tagger.Apply_POS as Apply_POS

'''
Reads the text corpus (dataset).
use_sentiment (true) for binary sentiment class (pos/neg)
'''
roots=[];
dicbased=[];
rulebased=[];

labelDict={"angry":1, "disgust":2, "fear":3, "happy":4, "sad":5, "surprise":6 }

uposorgoStrings=["অ","অঘা","অজ","অনা","আ","আড়","আন","আব","ইতি","ঊন","ঊনা","কদ","কু","নি","পাতি","বি","ভর","রাম","স",
                 "সা","সু","হা ","প্র","পরা","অপ","সম","নি","অনু","অব","নির","দুর","বি","অধি","সু","উৎ","পরি","প্রতি","অতি","অপি",
                 "অভি","উপ","আ","কার","দর","নিম","ফি","বদ","বে","বর","ব","কম","আম","খাস","লা","গর","ফুল","হাফ","হেড","সাব","হর"];

prottoi=["মুক্ত","গামী","গামীর","চারী","চারীর","বাহী","বাহীর","সমেত","মণ্ডলী","মণ্ডলীর",
	"কর্মা","কর্মার","কার্য","পূর্ব","কর","করা","করার","মালা","মালার","কীর্তি","কীর্তির","করণ","াংশ","গত","জনক","কণা","কণার","সহ",
	"মূলক","ভাবে","যোগ্য","ভিত্তিক","সূচক","কারী","কারীর","প্রবণ","প্রবণতা","প্রবণতার","সুলভ","পূর্ণ","তে","তা","কালীন","মত","ত্ব","পত্র",
	"মুখী","মুখীর","সম্পন্ন","ময়","সংক্রান্ত","সমৃদ্ধ","দার","ভরা","ভরার","গ্রস্ত","জনিত","জীবী","জীবীর","নীতি","নীতির","বিধি","বিধির","বাজ","বিমুখ",
	"শীল","বাদী","বাদীর","বান","গাছি","গাছির","নামা","নামার","জন","বহ","বার","ধারী","ধারীর","দায়ক","দাতা","দাতার","কাল","সূচি","সূচির","গার","ভোগী","ভোগীর", "ভাগী","ভাগীর", 
	"মাত্র","বিদ","বাদ","কণা","ভক্ত","ভুক্ত","ভুক্তি","ভুক্তির","ভুক্তিতে","পন্থা","পন্থী","পন্থীর","কামী","কামীর","যোগ","যোগে","বোধ","প্রতি",
	"কাজ","তম","বিষয়ক","সামগ্রী","সামগ্রীর","জাত","বাজি","বাজির","কবলিত","সূত্রে","জাত","বদ্ধ",
	"টা","টি","টু","টির","টার","টুকু","টুকুর","ধর্মী","ধর্মীর","যুক্ত","কৃত","ব্যাপী","ব্যাপীর","কেন্দ্রিক","বিরোধী","বিরোধীর","মুক্তি","মুক্তির",
	"রূপী","রূপীর","রূপে","রূপ","শালি","শালির","শালী","শালীর","প্রসূত","বাসী","বাসীর","প্রবাসী","প্রবাসীর","প্রাপ্ত","গোষ্ঠী","গোষ্ঠীর","বাচক","নির্ভর","স্বর",
	"সংলগ্ন","শাস্ত্র","মাফিক","স্বরূপ","কূল","সম্মত","সিদ্ধ","বৃন্দ","দান","সহ","বশত","ভর", 
	#i
	"ই","ও","এর","এ",
	#prottoi
	"য়ে","না","তি","তা","ওয়া","উয়া","উনি","উকা","ঈ","আও","অ"," রি"," তি"," তা","তে", " এ"," ঊ",
	" উরি"," উন্তি"," উক"," ইয়ে"," ইয়া"," ইলে"," আল",
	" আরী"," আরি"," আনো"," আনি"," আন"," আত"," আকু"," আইত"," আই"," আ"," অন্তি"," অন্ত"," অনি",
	" অনা"," অন"," অতি"," অতা"," অত"," অক",
	" োয়া","ুয়া","ুন্তি","ুনি","ুকা","ুকা","ু","ো","ে","ী","ীয়",
         #/*"ি",*/
         "য়",
	# prottoi-bangla(toddit)
	"ড়ে","ড়ি","ড়া","সে","সিয়","সা","লা","মি","মন্ত","ভরা","ভর","পারা","পানো","পনা","নি","তুত","তি","তা",
	"টে","টিয়া","টা","চে","কে","কিয়া","কা","ক","ওয়ালা","এল","এল","ঊ","উয়া","উড়িয়া","উড়","উলি","উরে",
	"উরিয়া","উক","ইয়াল","ইয়া","ই","আলো","আলি","আলা","আল","আরু","আরি","আর","আমো","আমি","আম",
	"আনো","আনি","আত","আচি","আচ","আইত","আই","আ","অড়","অল",
	#case
	"ের","য়েদেরকে","েদেরকে","দেরকে","য়েদের","েদের","য়েরা","ভাবে","দের","েরা","োর","ার","য়ের",
	#/*"কায়",*/
        "কার","িলা","িলা","িত","ান","কু","কা","তে","রা","য়","মি",
        #/*"সি","ড়ে",*/
        "কে","েই","র","া","ানু","বৃন্দ",
        #"য়","ব","ম","স",
	#article 
	"খানা","খানার","খানায়","খানি","খানির","গুলো","গুলো","গুলোর","গুলোয়","গুলোতে","গুলি","গুলির","গুলিতে","য়োন","খান","গুল","সমূহ","গণ",
	"টা","টি","টু","টির","টার","টুকু",
	"কা","তা","িল","িক",	"েক","েত",
        #/*"লি",*/
        "য়া","ায়","ড়ি","িস","ান",
	"হীনতা","হীন","বিহীন",
	#verb
	"ছিস", "ছে", "ছেন", "চ্ছি", "চ্ছ", "চ্ছে","চ্ছেন", 
	"েছি","েছ","েছে","েছেন",
	"াছি","াছ","াছে","াছেন",
	"িয়","ি",			
	"য়াছ","য়াছি","য়াছে","য়াছেন",
	"লাম","লেম","লুম","লেন", "ছিলে", "ছিল", "তো", 
	"তাম","তেম","তুম", "তেন", "ল",
	"বেন", "বো","বে","বি", "েন",
	"াম","েম","ুম",
	"িয়ে",
	"াই", 
	#normalization
	"িনি","নি","না","তেই","টাই","স্থ","ায়ন","াচ্ছন্ন",#/*"ক",*/
        "তে" #//normalize
        ];


must=["ই","ও","এর","এ","ের","য়েদেরকে","েদেরকে","দেরকে","য়েদের","েদের","য়েরা","ভাবে","দের","েরা","য়ের","রা","কে",
     "খানা","খানার","খানায়","খানি","খানির","গুলো","গুলো","গুলোর","গুলোয়","গুলোতে","গুলি","গুলির","গুলিতে","য়োন","খান","গুল",
     "সমূহ","গণ", 
     "মুক্ত","গামী","গামীর","চারী","চারীর","বাহী","বাহীর","সমেত","মণ্ডলী","মণ্ডলীর",
     "কর্মা","কর্মার","কার্য","পূর্ব","কর","করা","করার","মালা","মালার","কীর্তি","কীর্তির","করণ","াংশ","গত","জনক","কণা","কণার","সহ",
     "মূলক","ভাবে","যোগ্য","ভিত্তিক","সূচক","কারী","কারীর","প্রবণ","প্রবণতা","প্রবণতার","সুলভ","পূর্ণ","তে","তা","কালীন","মত","ত্ব","পত্র",
     "মুখী","মুখীর","সম্পন্ন","ময়","সংক্রান্ত","সমৃদ্ধ","দার","ভরা","ভরার","গ্রস্ত","জনিত","জীবী","জীবীর","নীতি","নীতির","বিধি","বিধির","বাজ","বিমুখ",
     "শীল","বাদী","বাদীর","বান","গাছি","গাছির","নামা","নামার","জন","বহ","বার","ধারী","ধারীর","দায়ক","দাতা","দাতার","কাল","সূচি","সূচির","গার","ভোগী","ভোগীর", "ভাগী","ভাগীর", 
     "মাত্র","বিদ","বাদ","কণা","ভক্ত","ভুক্ত","ভুক্তি","ভুক্তির","ভুক্তিতে","পন্থা","পন্থী","পন্থীর","কামী","কামীর","যোগ","যোগে","বোধ","প্রতি",
     "কাজ","তম","বিষয়ক","সামগ্রী","সামগ্রীর","জাত","বাজি","বাজির","কবলিত","সূত্রে","জাত","বদ্ধ",
     "টা","টি","টু","টির","টার","টায়","টুকু","টুকুর","ধর্মী","ধর্মীর","যুক্ত","কৃত","ব্যাপী","ব্যাপীর","কেন্দ্রিক","বিরোধী","বিরোধীর","মুক্তি","মুক্তির",
     "রূপী","রূপীর","রূপে","রূপ","শালি","শালির","শালী","শালীর","প্রসূত","বাসী","বাসীর","প্রবাসী","প্রবাসীর","প্রাপ্ত","গোষ্ঠী","গোষ্ঠীর","বাচক","নির্ভর","স্বর",
     "সংলগ্ন","শাস্ত্র","মাফিক","স্বরূপ","কূল","সম্মত","সিদ্ধ","বৃন্দ","দান","বশত","ভর",
     "সহ","ে","ের","েয়","ীয়","েক","িক","িতা","িণী","িনী","ী" ];
      
def stemmer( text ):
    WORD_PATTERN=" +|-|'|\b|\t|\n|\f|\r|\"|\'|#|%|&|~|@|<|>|\*|\+|\=|\.|,|\।৷|'|\)|\(|\{|\}|\[|\]|;|:|/|\\\\|-|_|—|–|”|“";
    st=re.split(WORD_PATTERN,text);
    stemWordList=[];
    roots.clear();
    roots.append(text);
    for str in st :
        if len(str)>0:
            dicbased.clear();
            rulebased.clear();
            stem = findStem(str);
            stemWordList.append(stem);


def findStem( word ):
    rootminstem="";
    dicbased.clear();
    temp="";
    rootminlen = 1000;
    dicbased.append(word);
    rightTrim(word);

    for i in range(0,len(dicbased)):
        temp=dicbased[i];
        length=len(temp);
        if temp not in roots:
            continue;
        if(length>=2 and length<rootminlen):
            rootminstem=temp;
            rootminlen=length;
    if len(rootminstem)>0:
        return rootminstem;
    rulebased.clear();
    rulebased.append(word);
    rightTrimMust(word);

    for i in range(0,len(rulebased)):
        temp=rulebased[i];
        length=len(temp);
        if length>=2 and length<rootminlen:
            rootminstem=temp;
            rootminlen=length;
    return rightTrimRa(rootminstem);
    
    

def rightTrim(word):
    if len(word)<3 :
        return;
    fraction=word;
    for str  in prottoi:
        if(word.endswith(str)):
            fraction = word[0:len(word)-len(str)];
            if(len(fraction)<2):
                continue;
            a="্";#0x09CD
            ch=fraction[len(fraction)-1];
            if ch==a:
                continue;
            if (len(word) != len(fraction)):
                dicbased.append(fraction);
                rightTrim(fraction);
    return;

def rightTrimMust( word ):
    if len(word)<3:
        return;
    fraction="";
    if word.endswith("য়") and word.endswith("ায়"):
        word=word[0:len(word)-1];
        rulebased.append(word);

    for i in range(0,len(must)):
        if word.endswith(must[i]):
            fraction=word[0:len(word)-len(must[i])];
            if len(fraction)<2:
                continue;
            a="্";#0x09CD
            ch=fraction[len(fraction)-1];
            if ch==a:
                continue;
            if len(word)!=len(fraction):
                rulebased.append(fraction);
                rightTrimMust(fraction);


def rightTrimRa(word):
    if len(word)<3:
        return word;
    fraction=word;
    if word.endswith("র"):
        if word.endswith("পুর"):
            return fraction;
        a="্"#0x09CD
        b="া"#0x09BE
        c="ৗ"#0x09D7
        ch=word[len(word)-2];
        if ch>=b and ch<=c and ch!=a:
            fraction=word[0:len(word)-1];
    return fraction;

def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []


    waste = '''’'()[]♥{}<>:,‒–—―….«»-‐‘’“”;/⁄␠·&@*\\•^¤¢$€£¥₩₪†‡°¡¿¬#
    №%‰‱¶′§~¨_|¦⁂☞∴‽※'"-\।()/'\"%#/@;:<>{}+=~|—‘’“”\.!?,`$^&*_+=
    abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
    ০১২৩৪৫৬৭৮৯🙂😀😄😆😅😂🤣😊☺😌😉😏😍😘😗😙😚🤗😳🙃😇😈😛😝😜😋🤤🤓😎🤑😒🙁
    ☹😞😔😖😓😢😢😭😟😣😩😫😕🤔🙄😤😠😡😶🤐😐😑😯😲😧😨😰😱😪😴😬🤥🤧🤒😷🤕😵🤢🤠
    🤡👿👹👺👻💀👽👾🤖💩🎃⚔👹\🐡🐟🐠🙂😀😂👍😐😡🏽✊☝🙌👍🏻👎🏻✌️🏻🤞🏻👌🏻🤙🏻🤘🏻🖕🏻☝
    🏻💅🏻👉🏻👈🏻👇🏻👆🏻💚💙❤💔💕💜💓💞💗💘💖💝💟'''

    with open(corpus_file, encoding='utf-8-sig') as f:
        for line in f:
            tokens = line.strip().split()
            temp = []
            for char in tokens[1:]:
                no_punct = ""
                for ch in char:
                    if ch not in waste:
                        no_punct = no_punct + ch

                if no_punct:
                    #print(no_punctuation)
                    no_punct = findStem(no_punct)
                    #print(no_punctuation)
                    if no_punct:
                        temp.append(no_punct)
                        
            documents.append(temp)
            if use_sentiment:
                # 2-class problem: positive vs negative
                #for line in data.label:
                #labels.append(labelDict[tokens[0]])
                labels.append(tokens[0])
    #for i in range(11):
     #   print(documents[i])
    return documents, labels



def read_stopwords(stopwords_file):
    stop_words = []
    with open(stopwords_file, encoding='utf-8-sig') as f:
        for line in f:
            stop_words.append(line.strip())

    return stop_words

def remove_stopwords(word_list, stopwords_list):
    processed_word_list = []

    for doc in word_list:
        temp = []
        for word in doc:
            if word not in stopwords_list:
                temp.append(word)

        processed_word_list.append(temp)

    return processed_word_list

# a dummy function that just returns its input
def identity(x):
    return x

def S_V_M_tf(train_data, train_label, test_data, test_label):
    #tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='utf-8', ngram_range=(1, 2))

    tfidf = TfidfVectorizer(preprocessor = identity, tokenizer = identity, min_df=2,
                            encoding='utf-8-sig', ngram_range=(1,2))

  #Using Linear SVC
    L_svc=Pipeline([('tfidf',tfidf),('ln_svc', LinearSVC())])
    L_svc.fit(train_data,train_label)
    testGuess=L_svc.predict(test_data)
    #classguess=L_svc.predict(predict)
    L_svc_cm = confusion_matrix(test_label, testGuess, labels=L_svc.classes_)
    prec, rec, f1, tureSum= precision_recall_fscore_support(test_label, testGuess)
    print("SVM with Tf-idf: ")
    #print("predicted class:", classguess)
    accuracy = accuracy_score(test_label, testGuess)
    print("Accuracy = ", accuracy)
    print("Confusion Matrix:")
    print(L_svc.classes_)
    print(L_svc_cm)
    print("Precision: ", prec)
    print("Recall:", rec)
    print("F1 Score: ", f1)
    print("\n") 
    cm_show = ConfusionMatrixDisplay(L_svc_cm, display_labels=['angry', 'happy', 'sad'])
    cm_show.plot()
    cm_show.ax_.set(
                title='Confusion Matrix for SVM', 
                xlabel='Predicted Emotion Class', 
                ylabel='Actual Emotion Class')
    plt.show()

# Fitting SVM to the Training set
    

def Multinomial_Naive_Bayes(trainDoc, trainClass, testDoc, testClass, tfIdf):

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfIdf:
        vec = TfidfVectorizer(preprocessor = identity,
                              tokenizer = identity, ngram_range=(1, 2))
   

    # Pipeline combines the vectorizers with a Naive Bayes classifier
    classifier = Pipeline( [('vec', vec),
                            ('cls', MultinomialNB())] )


    # Train the classifier and build a model using the training documents
    classifier.fit(trainDoc, trainClass)
    
    #saving the model
    from sklearn.externals import joblib
    joblib.dump(classifier, 'NB_model.pkl')
    
    '''
    #Loading the model:
    NB_spam_model = open('NB_model.pkl','rb')
    classifier = joblib.load(NB_spam_model)
    '''
    # Outputs Predicted class for the test set
    testGuess = classifier.predict(testDoc)
    
    try:
        #inputfile = open("input_nb.txt","r", encoding='utf-8')
        valDOC, valLBL = read_corpus('input_nb.txt', use_sentiment=True)
        predz = classifier.predict(valDOC)
        i=0
        for line in valDOC:
            print(line, predz[i])
            i+=1
        #inputfile.close()
    except:
        pass


    prec, rec, f1, tureSum= precision_recall_fscore_support(testClass, testGuess)
    #classguess = classifier.predict(predict)
    #print("predicted class:", classguess)

    # Simply calculates the accuracy score using the Gold Labels and Predicted Labels
    print("Naive Bayes with TF-idf:")
    accuracy = accuracy_score(testClass, testGuess)
    print("Accuracy = "+str(accuracy))
    print("Precision: ", prec)
    print("Recall:", rec)
    print("F1 Score: ", f1)
    #print("\n")

    # Showing the Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(testClass, testGuess, labels=classifier.classes_)
    print(classifier.classes_)
    print(cm)


# this is the main function but you can name it anyway you want
def main():
    # reads the whole dataset and separates documents(reviews) and labels(class/categories)
    # value of use sentiment determines whether to use binary(pos/neg) vs six way classification
    #DOC, LBL = read_corpus('rev.txt', use_sentiment=True)
    DOC, LBL = read_corpus('dataset3class.txt', use_sentiment=True)
    
    stop_words = read_stopwords('stopwords-bn.txt')
    DOC = remove_stopwords(DOC, stop_words)

    #print("length of doc =", len(DOC))

    ch = input('Do you want to apply POS-tagger? [Y/N]: ')

    if ch == 'Y' or ch == 'y':
        # This will call the HMM POS tagger. All the paths are specified in Config.py file
        model, vocab = HMM.start()

        # Apply POS Tagging on the above data set
        word_tag = HMM.decoding(model, vocab, DOC)

        # Reduce the features using particular set of POS (all_pos = False) or All pos (all_pos = True)
        # Go and check which POS is used, you can extend/reduce the list of POS to be considered
        DOC = Apply_POS.pos_feature(word_tag, all_pos = True)
    

    print("length of doc =", len(DOC))

    # Splits the dataset into training (90%) and test set(10%)
    split_point = int(0.90*len(DOC))
    trainDoc = DOC[:split_point]
    trainClass = LBL[:split_point]
    testDoc = DOC[split_point:]
    testClass = LBL[split_point:]

    print("Train data: ", len(trainDoc))
    print("Test data: ", len(testDoc))

    #for q in sorted(word2count.keys()):
    #    print(q, end = " ")

    # Calling the classifier (use the tf-idf/count feature/vectorizer)
    S_V_M_tf(trainDoc, trainClass, testDoc, testClass)
    #Multinomial_Naive_Bayes(trainDoc, trainClass, testDoc, testClass, tfIdf=True)
    
    


# program starts from here
if __name__ == '__main__':
    main()

