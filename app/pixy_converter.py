from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import pickle
import re
import difflib
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)

model=load_model("pixyModel.h5")
with open('pixyTokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
fillerListFilePath = "fillerList.csv"
accuracyFilePath = "accuracy.csv"
fillerData = pd.read_csv(fillerListFilePath, encoding='utf-8')
accuracyData = pd.read_csv(accuracyFilePath, encoding='utf-8')

fillerList = fillerData.apply(lambda col: [str(item) for item in col if item != ''], axis=0)
fillerList = fillerList.apply(lambda col: [x for x in col if x != 'nan'], axis=0)

accuracyList = accuracyData.apply(lambda col: [str(item) for item in col if item != ''], axis=0)

fillerTags = list(fillerList.index)
fillerTagIndexs = {}
for idx, word in enumerate(fillerTags, start=0):
    fillerTagIndexs[word] = idx

model_input_shape = model.layers[0].input_shape
maxLength = model_input_shape[1]

def predict_filler(model, tokenizer, sentence, fillerTagIndexs, sequence):
    predicted_probabilities = model.predict(np.array(sequence))[0]
    predicted_index = np.argmax(predicted_probabilities)
    predicted_word = [word for word, index in fillerTagIndexs.items() if index == predicted_index][0]
    return predicted_word

def insert_predicted_filler(model, tokenizer, sentence, fillerTagIndexs):
    if "[needFiller]" in sentence:
        sequence = tokenizer.texts_to_sequences([sentence])
        sequence = pad_sequences(sequence, maxlen=maxLength, padding='pre')
        predicted_tag = predict_filler(model, tokenizer, sentence, fillerTagIndexs, sequence)
        predicted_filler = random.choice(fillerList[predicted_tag])
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

def split_sentence_with_placeholder(sentence):
    needFiller_indices = [i for i, word in enumerate(sentence.split()) if "[needFiller]" in word]
    filled_sentences = []
    for index in needFiller_indices:
        filled_sentence = " ".join(sentence.split()[:index]) + " ".join(sentence.split()[index:]).replace("[needFiller]", "", 1)
        filled_sentences.append(filled_sentence)
    return filled_sentences

def merge_strings(str1, str2):
    s = difflib.SequenceMatcher(None, str1, str2)
    output = []
    for opcode, a0, a1, b0, b1 in s.get_opcodes():
        if opcode == 'equal':
            output.append(str1[a0:a1])
        elif opcode == 'insert':
            output.append(str2[b0:b1])
        elif opcode == 'delete':
            output.append(str1[a0:a1])
        elif opcode == 'replace':
            output.append(str2[b0:b1])
    return ''.join(output)

def merge_sentences(sentences):
    merged_sentence = sentences[0]
    sentences_count = len(sentences)
    for i in range(1,sentences_count):
        merged_sentence = merge_strings(merged_sentence,sentences[i])
    return merged_sentence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    request_sentence = data['request_sentence']
    request_sentences = split_sentences(request_sentence)
    combine_sentence=""
    for i, sentence in enumerate(request_sentences):
        if sentence.count('[needFiller]') >= 2:
            splitResults = split_sentence_with_placeholder(sentence)
            insertResults = []
            for splitCase in splitResults:
                insertResult = insert_predicted_filler(model, tokenizer, splitCase, fillerTagIndexs)
                insertResults.append(insertResult)
            combineResult = merge_sentences(insertResults)
            combine_sentence += combineResult
        else:
            combine_sentence += insert_predicted_filler(model, tokenizer, sentence, fillerTagIndexs)
        if i < len(request_sentences) - 1:
            combine_sentence += " "
    return jsonify({'combin_sentence': combine_sentence})

@app.route('/fillerInfo', methods=['GET'])
def get_filler_info():
    fillerTags = list(fillerList.index)
    fillers = fillerList.tolist()
    response = {
        "fillerTags": fillerTags,
        "fillers": fillers
    }
    return jsonify(response)

@app.route('/accuracyInfo', methods=['GET'])
def get_accuracy_info():
    accuracy_dict = accuracyList.to_dict(orient='list')
    return jsonify(accuracy_dict)

if __name__ == '__main__':
    app.run(debug=True)