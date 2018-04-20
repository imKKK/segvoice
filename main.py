import sys
from gmm import save_model, predict
from pprint import pprint
from seg import segment
if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('help: python main.py [task] [model|wav]')
        exit()

    task = sys.argv[1]

    if task == 'train':
        speech = sys.argv[2]
        save_model(speech, speech.replace('.wav', '.mdl'))
        print('done.')

    elif task == 'seg' or 'track':
        model = sys.argv[2]
        dialogue = sys.argv[3]
        rec = segment(model, dialogue, task)
        if task == 'track':
            pprint(rec)

    elif task == 'verify':
        model = sys.argv[2]
        speech = sys.argv[3]
        score = predict(model, speech)
        print('yes') if score < 48 else print('no')
