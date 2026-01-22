
data_path = '/Users/crispinm.lang/Documents/CO5_python_labs/Continued_training/shakespeare.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Character size: ', len(text))
print('Vocabulary size: ', vocab_size)
