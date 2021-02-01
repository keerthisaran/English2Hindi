from models import Enc2DecWithAttention
CONFIGS={
    'ENC_DEC_1':  { 'model' : Enc2DecWithAttention,
                    'params': {
                                'lr': None,
                                'enc_hidden': 256,
                                'dec_hiddden': 256,
                                'optim':'adam'
                            } 

                }
}

