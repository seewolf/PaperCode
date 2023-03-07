import pandas as pd
import spacy

def to_CSV(path,csvPath):
    f=open(path, 'r', encoding='utf-8', newline='\n')
    lines=f.readlines()
    f.close()
    texts=[]
    labels=[]
    for i in range(0,len(lines)):
        label, text = lines[i].split('\t', 1)
        texts.append(text)
        labels.append(int(label))
    df=pd.DataFrame({'text':texts,'label':labels})
    df.to_csv(csvPath,index=True, index_label="id")

if __name__ == '__main__':
    to_CSV('./train.txt','./train.csv')
    to_CSV('./dev.txt','./dev.csv')
    to_CSV('./test.txt','./test.csv')

    



