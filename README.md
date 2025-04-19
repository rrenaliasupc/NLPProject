# Intent Extraction for Catalan Smart Home Commands

*Made by Ricard Renalias and Eric Roy as the final project
for Natural Language Processing, an elective course for the
ML and Cybersecurity for Internet-Connected Devices MSc.*

## Overview

This repository contains a NER that finds the needed tags for a voice
assistant to determine what actions it should do.
- Dataset generation tools.
- Speech to text using Whisper to have "real" (i.e. bad) transcriptions.
- Scripts to train and infer the model, based on BERT.

## Repository distribution

The files are structured as follows:
- In `dataset` you will find everything related to build the dataset or expand it.
- In `doc` you will find the report and theirs sources.
- In `build` you will find the model (due to its size, not included).
- In `src` you will find the source code to train and test the model.

## Run the demo

Install dependencies:

```sh
python3 -m pip install -r requirements.txt
```

### Prepare dataset

Optional: run `dataset/expand_dataset.py` and the Jupyter notebook in `dataset/home_assistant_expansion`.
Otherwise you can use the already included dataset.

### Create the model

Run `src/train.py`.

### Try it!

Run `src/main.py`. You can record yourself and press enter to stop. Then it will
transcribe your voice and infer the model.

Tip! If you don't want to test the slow-to-load and not-made-by-us Whisper, you
can just run something like:

```sh
python3 src/main.py --text "Engega el llum del labavo"
```

Be sure to use quotes so the entire text is in a single argv.

## More information

For more information, please refer to the project report. You can find
a build of it, if not already in that directory, on the
[GitHub release page](https://github.com/rrenaliasupc/NLPProject/releases).

## Contributing

This project is licensed under the MIT License. This is a close project that
does not accept contributions per se, but we're more than happy for you to use
this tool to build something more meaningful.
