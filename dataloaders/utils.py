from transformers import BertTokenizer,RobertaTokenizer
import numpy as np
from operator import itemgetter
import spacy 
import en_core_web_sm
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class Tokenizer4Bert:
    def __init__(self, max_seq_len,pretrained_bert_name):
        if pretrained_bert_name=='bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        elif pretrained_bert_name in ['roberta-base' ,'roberta-large']:
            self.tokenizer=RobertaTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        self.tw_preprocessor=twitter_preprocessor()
    def __call__(self,text):
        text=' '.join(self.tw_preprocessor(text))
        return self.tokenizer(text,truncation=True,padding="max_length",max_length=self.max_seq_len)


def preprocess(info,tokenizer):
    
    text=info['text'] if 'text' in info.keys() else info['sentence']
    
    label=info['label']
    
    temp=tokenizer(text)
    
    return {
            "text":text,
            "label":label,
            "input_ids":temp['input_ids'],
            "token_type_ids":temp['token_type_ids'] if "token_type_ids"in temp.keys() else np.zeros(len(temp["attention_mask"])),
            "attention_mask":temp['attention_mask']
            }
    

def go_emotion_preprocess(info,tokenizer):
    text=info['text']
    id=info['id']
    labels=info['labels']
    temp=tokenizer(text)
    return {
        "text":text,
        "labels":labels,
        "input_ids":temp['input_ids'],
        "token_type_ids":temp['token_type_ids'] if "token_type_ids"in temp.keys() else np.zeros(len(temp["attention_mask"])),
        "attention_mask":temp['attention_mask']
        }
    
def semeval18_preprocess(info,tokenizer):
    label_list=['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']
    text=info["Tweet"]
    id=info["ID"]
    labels=np.array(itemgetter(*label_list)(info)).astype("int32")
    temp=tokenizer(text)
    return {
        "text":text,
        "labels":labels,
        "input_ids":temp['input_ids'],
        "token_type_ids":temp['token_type_ids'] if "token_type_ids"in temp.keys() else np.zeros(len(temp["attention_mask"])),
        "attention_mask":temp['attention_mask']
    }
    

    
        




