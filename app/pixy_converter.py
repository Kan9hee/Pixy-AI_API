from flask import Flask, request, jsonify
import numpy as np
import pickle
import re
import nltk
from nltk import sent_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model=load_model("pixyModel.h5")
with open('pixyTokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
fillers = [", um,",", well,",", you know,","You see,","So,","I guess","I suppose"," You know what I mean?"]

filler_index = {}
for idx, word in enumerate(fillers, start=0):
    filler_index[word] = idx

def predict_filler(model, tokenizer, sentence, filler_index, max_length=100):
    sequence = tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_length, padding='pre')
    predicted_probabilities = model.predict(np.array(sequence))[0]
    predicted_index = np.argmax(predicted_probabilities)
    predicted_word = [word for word, index in filler_index.items() if index == predicted_index][0]
    return predicted_word

def insert_predicted_filler(model, tokenizer, sentence, filler_index, max_length=100):
    if "[needFiller]" in sentence:
        sequence = tokenizer.texts_to_sequences([sentence])
        sequence = pad_sequences(sequence, maxlen=max_length, padding='pre')
        predicted_filler = predict_filler(model, tokenizer, sentence, filler_index, max_length)
        if ", [needFiller]" in sentence:
            completed_sentence = sentence.replace(", [needFiller]", predicted_filler)
        elif " [needFiller]" in sentence:
            completed_sentence = sentence.replace(" [needFiller]", predicted_filler)
        else:
            completed_sentence = sentence.replace("[needFiller]", predicted_filler)
        return completed_sentence
    else:
        return sentence
    
def split_sentences(sentence):
    sentences = re.split(r'(?<=[.?!])\s+', sentence)
    result = []
    for i, s in enumerate(sentences):
        if "[needFiller] [needFiller]" in s:
            s = s.replace("[needFiller] [needFiller]", "[needFiller]")
            result.append(s)
            if i > 0:
                result[-2] += " [needFiller]"
        else:
            result.append(s)
    return result

def split_sentences(sentence):
    # [needFiller]가 나타나는 위치 찾기
    needFiller_indices = [i for i, word in enumerate(sentence.split()) if "[needFiller]" in word]
    filled_sentences = []
    # 모든 경우의 수에 대해 문장 생성
    for index in needFiller_indices:
        # [needFiller]를 제거한 문장 생성
        filled_sentence = " ".join(sentence.split()[:index]) + " ".join(sentence.split()[index:]).replace("[needFiller]", "", 1)
        filled_sentences.append(filled_sentence)
    return filled_sentences

def grammar_check(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        grammar_checked_sentence = " ".join(word for word, tag in tagged_words)
        corrected_sentences.append(grammar_checked_sentence)
    corrected_text = " ".join(corrected_sentences)
    return corrected_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    request_sentence = data['request_sentence']
    request_sentences = split_sentences(request_sentence)
    combin_sentence=""
    for i, sentence in enumerate(request_sentences):
        combin_sentence += insert_predicted_filler(model, tokenizer, sentence, filler_index)
        if i < len(request_sentences) - 1:
            combin_sentence += " "
    completed_sentence = grammar_check(combin_sentence)
    return jsonify({'completed_sentence': completed_sentence})

if __name__ == '__main__':
    app.run(debug=True)