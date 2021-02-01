
from sklearn.model_selection import train_test_split
from config import CONFIGS

def train(model_str):
    model_class=CONFIGS[model_str]['model']
    model_struct=CONFIGS[model_str]['struct']
    params=CONFIGS[model_str]['params']

    model=model_class(**model_struct)
    train_generator=None
    valid_generator=None
    epochs=2000
    batch_size=256
    model.compile(params)
    model.fit(train_generator=train_generator,
             valid_generator=valid_generator,
             epochs=epochs,
             batch_size=batch_size,
             metrics=['accuracy'])
    
    model.save(path=None)





    

