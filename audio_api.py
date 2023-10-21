import os
import librosa
import numpy as np
from flask import Flask, request, jsonify, render_template
# Import the Streamlit library

import numpy as np
import os, sys, getopt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import json
from chromagram import compute_chroma
import hmm as hmm

# Title for your app



def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("data/chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord == "N":
            continue
        templates.append(templates_json[chord])

    return templates


def get_nested_circle_of_fifths():
    chords = [
        "N",
        "G",
        "G#",
        "A",
        "A#",
        "B",
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "Gm",
        "G#m",
        "Am",
        "A#m",
        "Bm",
        "Cm",
        "C#m",
        "Dm",
        "D#m",
        "Em",
        "Fm",
        "F#m",
    ]
    nested_cof = [
        "G",
        "Bm",
        "D",
        "F#m",
        "A",
        "C#m",
        "E",
        "G#m",
        "B",
        "D#m",
        "F#",
        "A#m",
        "C#",
        "Fm",
        "G#",
        "Cm",
        "D#",
        "Gm",
        "A#",
        "Dm",
        "F",
        "Am",
        "C",
        "Em",
    ]
    return chords, nested_cof


def find_chords(
    x: np.ndarray,
    fs: int,
    templates: list,
    chords: list,
    nested_cof: list = None,
    method: str = None,
    plot: bool = False,
):
    """
    Given a mono audio signal x, and its sampling frequency, fs,
    find chords in it using 'method'
    Args:
        x : mono audio signal
        fs : sampling frequency (Hz)
        templates: dictionary of chord templates
        chords: list of chords to search over
        nested_cof: nested circle of fifth chords
        method: template matching or HMM
        plot: if results should be plotted
    """

    # framing audio, window length = 8192, hop size = 1024 and computing PCP
    nfft = 8192
    hop_size = 1024
    nFrames = int(np.round(len(x) / (nfft - hop_size)))
    # zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(nfft))
    xFrame = np.empty((nfft, nFrames))
    start = 0
    num_chords = len(templates)
    chroma = np.empty((num_chords // 2, nFrames))
    id_chord = np.zeros(nFrames, dtype="int32")
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    # step 1. compute chromagram
    for n in range(nFrames):
        xFrame[:, n] = x[start : start + nfft]
        start = start + nfft - hop_size
        timestamp[n] = n * (nfft - hop_size) / fs
        chroma[:, n] = compute_chroma(xFrame[:, n], fs)

    if method == "match_template":
        # correlate 12D chroma vector with each of
        # 24 major and minor chords
        for n in range(nFrames):
            cor_vec = np.zeros(num_chords)
            for ni in range(num_chords):
                cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))
            max_cor[n] = np.max(cor_vec)
            id_chord[n] = np.argmax(cor_vec) + 1

        # if max_cor[n] < threshold, then no chord is played
        # might need to change threshold value
        id_chord[np.where(max_cor < 0.8 * np.max(max_cor))] = 0
        final_chords = [chords[cid] for cid in id_chord]

    elif method == "hmm":
        # get max probability path from Viterbi algorithm
        (PI, A, B) = hmm.initialize(chroma, templates, chords, nested_cof)
        (path, states) = hmm.viterbi(PI, A, B)

        # normalize path
        for i in range(nFrames):
            path[:, i] /= sum(path[:, i])

        # choose most likely chord - with max value in 'path'
        final_chords = []
        indices = np.argmax(path, axis=0)
        final_states = np.zeros(nFrames)

        # find no chord zone
        set_zero = np.where(np.max(path, axis=0) < 0.3 * np.max(path))[0]
        if np.size(set_zero) > 0:
            indices[set_zero] = -1

        # identify chords
        for i in range(nFrames):
            if indices[i] == -1:
                final_chords.append("NC")
            else:
                final_states[i] = states[indices[i], i]
                final_chords.append(chords[int(final_states[i])])

    if plot:
        plt.figure()
        if method == "match_template":
            plt.yticks(np.arange(num_chords + 1), chords)
            plt.plot(timestamp, id_chord, marker="o")

        else:
            plt.yticks(np.arange(num_chords), chords)
            plt.plot(timestamp, np.int32(final_states), marker="o")

        plt.xlabel("Time in seconds")
        plt.ylabel("Chords")
        plt.title("Identified chords")
        plt.grid(True)
        plt.show()

    return timestamp, final_chords

app = Flask(__name__)

# Create a directory to store uploaded audio files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('audio_upload.html')

@app.route('/process_realtime_audio', methods=['POST'])
def process_audio():

    # Check if the file was sent in the POST request
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})
    
    try:
        # Save the uploaded file to a temporary location

        file = request.files['audio']
        file.save(file.filename)

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Load the audio file and obtain the mono audio signal and sampling frequency
        print("input getting")
        audio_data, sample_rate = librosa.load(file.filename, sr=None, mono=True)
        # audio_data,sample_rate = read(file.filename)
        print("input not getting")
        print(audio_data)
        print(sample_rate)
        method = "match_template"
        plot = False
        has_method = False
        
        print("entering")
        # x = audio_data[:, 0] if len(audio_data.shape) else audio_data
        # get chords and circle of fifths
        print("exiting")
        chords, nested_cof = get_nested_circle_of_fifths()
        # get chord templates
        print("complete1")
        templates = get_templates(chords)
         # find the chords
        print("complete2")
        if method == "match_template":
            timestamp, final_chords = find_chords(
                audio_data, sample_rate, templates=templates, chords=chords, method=method, plot=plot
             )
        else:
            timestamp, final_chords = find_chords(
                audio_data,
                sample_rate,
                templates=templates,
                chords=chords[1:],
                nested_cof=nested_cof,
                method=method,
                plot=plot,
            )
                # print chords with timestamps
        print("compley3s")
        print("Time (s)", "Chord")
        print(final_chords)
        final_ch = ""
        for i in range(len(final_chords)):
             if(final_chords[i] == 'A'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'A#'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'B'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'C'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'C#'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'D'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'D#'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'E'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'E#'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'G'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
             if(final_chords[i] == 'G#'):
                 print("The Identified Chord is")
                 print(final_chords[i])
                 final_ch = final_chords[i]
                 break
        print("over")
        print('finally' + final_ch)
        
        # Return the mono audio signal and sampling frequency
        return jsonify({
            'x': audio_data.tolist(),
            'fs': sample_rate,
            'final_chord' : final_ch
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
