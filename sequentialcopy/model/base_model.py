from tensorflow import keras

class CopyModel(keras.Model):
    def __init__(self):
        super(CopyModel, self).__init__()   
            
    def fit(self, x, y, epochs, batch_size, verbose):
        return super(CopyModel,self).fit(x,y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def compile(self):
        super(CopyModel, self).compile()
        