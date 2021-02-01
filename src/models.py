from abc import ABC, abstractmethod
import torch
from tqdm import tqdm,trange

class Engine(ABC):

    @abstractmethod
    def fit(self,kwargs):
        pass

    @abstractmethod
    def save(self,path):
        pass

    @abstractmethod
    def predict(self,path):
        pass

class PytorchEngine():

    def __init__(self,model):
        self.model=model
        pass
    
    def compile(self,optimizer='adam', loss=None, metrics=None):
        self.optim=torch.optim.Adam(self.model.parameters(),lr=0.005)
        self.loss=torch.nn.NLLLoss()
        self.metrics=metrics
        

    def fit(self,
             train_generator,
             train_valid_split=0.9,
             epochs=5,
             test_generator=None,
             batch_size=64,
             metrics=None):
        
        train_generator=train_generator
        total_samples=train_generator.total_samples
        nbatches=total_samples//batch_size
        total_loss=0
        mean_loss_array=[]
        for epoch in range(epochs):
            with trange(nbatches) as t:
                for batch_i in t:
                    X_batch,Y_batch=train_generator.get_batch(batch_size)
                    maxlens=[len(Y) for Y in Y_batch]
                    batch_pred=self.model(X_batch,maxlens,Y_batch)
                    #batch_pred should be tensor of shape (N,outdim), Y_batch (N) with int range 0,output_dim-1
                    batch_loss=self.get_batchloss(batch_pred,Y_batch)
                    self.optim.zero_grad()
                    batch_loss.backward()
                    self.optim.step()
                    total_loss+=batch_loss.item()
                    
                    mean_loss_array.append(total_loss/(epoch*nbatches+batch_i+1))
                    t.set_postfix(epoch=epoch,loss=batch_loss.item(),cur_mean_loss=mean_loss_array[-1])
        return mean_loss_array

    def get_batchloss(self,batch_pred,Y_batch):
        batch_loss=torch.zeros((1,),dtype=torch.float32)
        for pred,act in zip(batch_pred,Y_batch):
            batch_loss+=self.loss(pred,act)
        
        return batch_loss


class Enc2DecWithAttention(torch.nn.Module):

    def __init__(self,input_dim,encoder_decoder_hidden_dim,output_dim,attention_layers_dim=None):
        super().__init__()
        self.input_dim=input_dim
        self.encoder_decoder_hidden_dim=encoder_decoder_hidden_dim
        self.output_dim=output_dim

        self.encoder=torch.nn.GRU(self.input_dim,self.encoder_decoder_hidden_dim)

        self.attention_layer=torch.nn.Linear(2*encoder_decoder_hidden_dim,1)
        self.attention_softmax=torch.nn.Softmax(dim=-1)

        self.decoder=torch.nn.GRU(self.output_dim+self.encoder_decoder_hidden_dim,encoder_decoder_hidden_dim)

        self.output_layer=torch.nn.Linear(self.encoder_decoder_hidden_dim,self.output_dim)        
        self.output_softmaxlog=torch.nn.LogSoftmax(dim=-1)

        self.engine=PytorchEngine(self)


    def forwardsample(self,X,max_len,ground_truth):
        '''
        X is a torch.tensor with shape [None,input_shape(English Vocabulary)] and dtype torch.float32
        Y is a 1d tensors with shape [None,] and dtype torch.long

        '''
        
        
        enc_hiddens,last_enc_hidden=self.encoder(X.unsqueeze(1))
        prev_dec_hidden=last_enc_hidden
        prev_dec_input=torch.zeros(1,1,self.output_dim)

        enc_seq_len=enc_hiddens.shape[0]
        outputs=[]
        for i in range(max_len):
            #enc_hiddens shape is (None,1,encoder_decoder_hidden)
            attention_input=torch.cat([ prev_dec_hidden.repeat(enc_seq_len,1,1),
                                        enc_hiddens],dim=-1)
            attention_logits=self.attention_layer(attention_input.squeeze(1))
            attention_logits=torch.nn.ReLU()(attention_logits)

            attention_wts=self.attention_softmax(attention_logits.squeeze(-1))
    
            enc_input=torch.matmul(attention_wts,enc_hiddens.squeeze(1))
            dec_input=torch.cat([enc_input,prev_dec_input.view(-1)])

            _,prev_dec_hidden,=self.decoder(dec_input.view(1,1,-1),prev_dec_hidden)

            output=self.output_layer(prev_dec_hidden.view(-1))
            output=self.output_softmaxlog(output)
            with torch.no_grad():
                prev_dec_input=torch.zeros_like(prev_dec_input,dtype=torch.float32)
                argmax_output=torch.argmax(output) if ground_truth is None else ground_truth[i]
                prev_dec_input[0,0,argmax_output]=1.

            outputs.append(output)
        return torch.stack(outputs)
    
    def forward(self,X_batch,maxlens,ground_truth=None):
        batch_pred=[]
        for i,(X,maxlen) in enumerate(zip(X_batch,maxlens)):
            ground_truth_i=ground_truth[i] if ground_truth else None
            Y_hat=self.forwardsample(X,maxlen,ground_truth_i)
            batch_pred.append(Y_hat)
        
        return batch_pred
    
    def fit(self,*args,**kwargs):
        return self.engine.fit(*args,**kwargs)
    
    def compile(self,*args,**kwargs):
        return self.engine.compile(*args,**kwargs)



if __name__ == "__main__":

    X=[torch.zeros(7,24,dtype=torch.float32) for _ in range(10)]
    Y=[torch.zeros(6,dtype=torch.long) for _ in range(10)]

    net=Enc2DecWithAttention(24,256,9)
    maxlens=[len(w) for w in Y]
    print(net.forward(X,maxlens,Y))
    pass
            











