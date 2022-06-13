import spacy
import string
# from spacy.lang.en import English
from pathlib import Path
import pickle


spacy.prefer_gpu()
nlp = spacy.load('en_core_web_lg')
nlp.max_length = 3000000
# test = "this is someting I need to test out. how are you. I'm doing just fine thanks for asking"

def get_sentences(sentences):
    new_list = []
    list_len = []
    for sentence in sentences:
        sent_clean = " ".join(str(sentence).split()) # remove long whitespaces
        sent_clean = sent_clean.translate(str.maketrans('', '', string.punctuation)).lower()  # remove punctuation
        sent_clean = nlp(sent_clean) # back to Spacy object - OPTIMIZATION HERE ???
        lenght = len([token.text for token in sent_clean])
        if lenght>4:
            # print([token.text for token in sent_clean])
            # print(lenght)
            # tmp_str = " ".join(str(sentence).split()) # remove long whitespaces
            # tmp_str = tmp_str.translate(str.maketrans('','',string.punctuation)).lower() #remove punctuation
            new_list.append(str(sent_clean)) #return String not Spacy Token object
            list_len.append(lenght)
    return (new_list,list_len)

# mypath = Path('/data/txtFiles/GutJustOne/')
mypath = Path('/data/txtFiles/GutSubset')
# mypath = Path('/data/txtFiles/Gut1k')
# mypath = Path('/data/txtFiles/Gutenberg/txt')
fileNames = []

def translate_to_sentences(path):
    for file in path.iterdir():
        # print(file.name)
        if file.is_file() and file.suffix == '.txt': # list of all files in the directory
            fileNames.append(path / file.name)

    master_sentence = []
    master_len = []
    # print(len(fileNames))
    idx = 1
    for file in fileNames: #for each file in the directory
        with open(file,"r") as file:
            text = file.read().replace('\n\n'," ").replace("\n", " ").replace("\t"," ") # is it needed if get_sentences fun contains cleanup ??
            doc = nlp(text) #create spacy token
            sentences = doc.sents # split text into sentences ?
            sentence_list,sentence_len = get_sentences(sentences=sentences) # just for the text / sentences clean up and remove too short sntences.
            # sentence_list => list of lists containing all sentences
            # sentence_len => list of numbers of sentence lenghts

            master_sentence+=sentence_list
            master_len+=sentence_len

            print(f"Proc file {idx} // {len(fileNames)}.Sent len {len(master_sentence)} len of len {len(master_len)} -- {file.name}")

            idx+=1
            if idx%100 == 0:
                with open(path / f'1_master_sentence_{idx}.pkl', 'wb') as f:
                    pickle.dump((master_sentence,master_len), f)
                    print(f'just pickled 1_master_sentence_{idx}.pkl')

    print(len(master_sentence),len(master_len))

    with open(path / '1_master_sentence_ALL.pkl','wb') as f:
        pickle.dump((master_sentence,master_len),f)
        print(f'just pickled 1_master_sentence_ALL.pkl')


translate_to_sentences(mypath)



# doc = nlp(test)
# sentences = doc.sents
# print(sentences)
# sentence_list = get_sentences(sentences=sentences)


# with open(path/'master_sentence.pkl','rb') as f:
#     master_sentence = pickle.load(f)

# text = "this is rally frustrating.          I'm stuck            here with this for few days. Unable           to procced My san is comming back from school shortly. Alice and John Kowalski are not together now."




# print(len(sentence_list))


# for sent in sentence_list[:10]:
#     print(sent)



# text = "this is rally frustrating. I'm stuck here with this for few days. Unable to procced My san is comming back from school shortly. Alice and John Kowalski are not together now."

# nlp_simple = English()
# nlp_simple.add_pipe('sentencizer')

# nlp = spacy.load('en_core_web_sm')

# text = 'My first birthday was great. My 2. was even better.'
#
# for nlp in [nlp_simple, nlp_better]:
#     for i in nlp(text).sents:
#         print(i)
#     print('-' * 20)


# doc = nlp(text)
#
# sentences = list(doc.sents)
# print(sentences)
# # print(type(sentences[0])) #<class 'spacy.tokens.span.Span'>
# sentence = sentences[3]
# print(sentences[3])
# entit = list(doc.ents)
# print(entit)