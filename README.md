# Speaker ID Model

Real-time Speaker identification model using SkLearn and KNN. Potentially could be used in
tandem with RNBO. Implementation of [this MATLAB article](https://www.mathworks.com/help/audio/ug/speaker-identification-using-pitch-and-mfcc.html)

## Preparing The Dataset

Audio training data for each speaker should be placed in the `files` folder as such...

```
files
├── david
    ├── 1.wav
    ├── 2.wav
    └── ...
├── max
└── sam
```

Audio files in each of the speaker folders should be in the format of 1 second wav files.
You can easily convert a super long wav file of speach into 1 second chunks using
ffmpeg...

```
ffmpeg -i input-speech.wav -f segment -segment_time 1 out%01d.wav
rm input-speech.wav
```

## Performing The Feature Extraction

Once the audio files are prepared in the format above, we can extract the necessary
features to a csv file in order to train our model

First, in `__main__.py`, replace the files in line 23 with your defined drives.

Next, if you haven't extracted features yet, uncomment line 80. This will take quite some
time depending on how much audio data you've supplied for training the model.

If you want to test the model on validation set data before running the model, uncomment
line 88.

## Running The Model on Real-Time Input

After running the feature extraction, an audio callback stream will begin using your
default audio input. You can now see how well the model predicts speakers by providing
new audio input from the provided speakers!

At the moment, predictions are printed in the console using numbers that correspond to
each speaker.
