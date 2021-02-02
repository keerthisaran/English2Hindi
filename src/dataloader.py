import xml
import pandas as pd
import numpy as np
import torch
import copy

class Eng2HindiDataloader():
    
    def __init__(self,xml_filepath,eng_minchar_freq=0.98,hind_minchar_freq=0.98):
        self.df=self.get_df(xml_filepath)
        self.total_samples=self.df.shape[0]
        self.english_chars=set(self.df['english'].apply(str).sum())
        self.english2index={c:i for i,c in enumerate(self.english_chars)}
        self.hindi_chars=set(self.df['hindi'].apply(str).sum())
        self.hindi2index={c:i for i,c in enumerate(self.hindi_chars)}
        self.index2hindi={i:c for i,c in enumerate(self.hindi_chars)}
        
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
            try:
                eng_tensor=self.convert_word2index(row['english'],'english')
                hind_tensor=self.convert_word2index(row['hindi'],'hindi',False)
                hind_tensor=torch.cat([hind_tensor,torch.zeros(1,dtype=hind_tensor.dtype)])
                X.append(eng_tensor)
                Y.append(hind_tensor)
            except:
                print('skipping {english}-{hindi}'.format(english=row['english'],hindi=row['hindi']))
        return X,Y
    
    def get_validgenerator(self,xml_filepath):
        valid_generator=copy.deepcopy(self)
        df=valid_generator.get_df(xml_filepath)
        valid_generator.df=df

        valid_generator.total_samples=df.shape[0]
        return valid_generator

        
    def get_df(self,xml_filepath):
        
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_filepath).getroot()
        full_train_df_dict={}
        full_train_df_dict['english']=[]
        full_train_df_dict['hindi']=[]
        for child in root:
            english_words=child[0].text.split()
            hindi_words=child[1].text.split()
            if len(english_words)!=len(hindi_words) or any([not x.isalpha() for x in english_words]):
                print('skipping',english_words,hindi_words)
                continue
            
            full_train_df_dict['english'].extend(map(lambda x:str(x).upper(),english_words))
            full_train_df_dict['hindi'].extend(hindi_words)


        return pd.DataFrame(full_train_df_dict)





    

if __name__ == "__main__":
    
    train_generator=Eng2HindiDataloader('inputs/train_data.xml')
    valid_generator=train_generator.get_validgenerator('inputs/test_data.xml')

    # train_generator=train_loaderbuilder.get_dataloader()
    # test_generator=test_loaderbuilder.get_dataloader()

    from models import Enc2DecWithAttentionBidir,PytorchEngine
    net=Enc2DecWithAttentionBidir(train_generator.len_english_chars,128,train_generator.len_hindi_chars)
    engine=PytorchEngine(net)
    engine.compile()
    engine.fit(train_generator,test_generator=valid_generator)





        