from tensorflow.keras.models import load_model

lr_rate = [0.00001, 0.0001, 0.0005, 0.0010, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
classes = ['dogs', 'cats']
filename = classes[1]+'_vs_'+ classes[0] + '_' + str(lr_rate[0]) +'.h5'
model = load_model(filename)