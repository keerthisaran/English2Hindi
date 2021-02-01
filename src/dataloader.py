import xml
import pandas as pd
import numpy as np
import torch

class Eng2HindiDataloader():
    
    def __init__(self,df,eng_minchar_freq=0.98,hind_minchar_freq=0.98):
        self.df=df

        self.english_chars=set(df['english'].apply(str).sum())
        self.english2index={c:i for i,c in enumerate(self.english_chars)}

        self.hindi_chars=set(df['hindi'].apply(str).sum())
        self.hindi2index={c:i for i,c in enumerate(self.hindi_chars)}
        self.index2hindi={i:c for i,c in enumerate(self.hindi_chars)}
        self.total_samples=df.shape[0]
        self.len_english_chars=len(self.english_chars)
        self.len_hindi_chars=len(self.hindi_chars)


    def convert_word2index(self,word, language='english',onehot_format=True):

        if language=='english':
            chars=self.english_chars
            chars2index=self.english2index
        else:
            chars=self.hindi_chars
            chars2index=self.hindi2index


        len_chars=len(chars)
        seq_len=len(word)


        onehot_id=[chars2index[c] for c in word]
        if onehot_format:
            ret=np.zeros((seq_len,len_chars),dtype=np.float32)
            
            ret[range(seq_len),onehot_id]=1.
        else:
            ret=np.array(onehot_id,dtype=np.long)
        ret=torch.from_numpy(ret)
        return ret

        
    def get_batch(self,batch_size):
        batch_df=self.df.sample(batch_size)

        X=[]
        Y=[]

        for row_num,row in batch_df.iterrows():

            eng_tensor=self.convert_word2index(row['english'],'english')
            hind_tensor=self.convert_word2index(row['hindi'],'hindi',False)
            hind_tensor=torch.cat([hind_tensor,torch.zeros(1,dtype=hind_tensor.dtype)])
            X.append(eng_tensor)
            Y.append(hind_tensor)
        
        return X,Y
        

class Eng2HindiDataloaderBuilder():

    def __init__(self,train_filepath='/home/ubuntu/EnglishToHindi/inputs/train_data.xml'):
        import xml.etree.ElementTree as ET
        root = ET.parse(train_filepath).getroot()
        full_train_df_dict={}
        full_train_df_dict['english']=[]
        full_train_df_dict['hindi']=[]
        for child in root:
            english_words=child[0].text.split()
            hindi_words=child[1].text.split()
            if len(english_words)!=len(hindi_words):
                print('skipping',english_words,hindi_words)
                continue
            full_train_df_dict['english'].extend(english_words)
            full_train_df_dict['hindi'].extend(hindi_words)
        
        self.full_train_df=pd.DataFrame(full_train_df_dict)


    def get_dataloader(self):
        eng2hind_dataloader=Eng2HindiDataloader(self.full_train_df)
        return eng2hind_dataloader

    

if __name__ == "__main__":
    
    x=Eng2HindiDataloaderBuilder()
    train_generator=x.get_dataloader()
    from models import Enc2DecWithAttention
    net=Enc2DecWithAttention(train_generator.len_english_chars,256,train_generator.len_hindi_chars)
    net.compile()
    net.fit(train_generator)





        