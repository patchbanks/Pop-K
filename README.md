# Pop-K

Pop-K is a generative MIDI model that creates pop melodies in C major and A minor. Built on RWKV-4, the architecture employs a pseudo-linear attention mechanism for efficient training and fast inference. The model was trained on ~300k 8-bar melody samples composed of augmented bass, chords and vocal/lead melodies.


## Model

Download the checkpoint from Hugging Face:

[![Pop_K](https://img.shields.io/badge/Pop_K-Hugging%20Face%20-blue)](https://huggingface.co/patchbanks/Pop-K/tree/main)

## Inference

Use `colab_run.ipynb` to run the model with Gradio or generate MIDI files in bulk.

## Examples

Preview MIDI outputs in data/midi_output_examples.zip.

## Dataset

The Pop-K MIDI Dataset is closed-source and not publicly available for direct download. However, more information about the dataset, including details on access and usage, can be found at the following link:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14791511.svg)](https://doi.org/10.5281/zenodo.14791511)


