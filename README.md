# deep-waifu-model

This is a part of the [DeepWaifu](https://blog.chronologic.network/no-waifu-no-laifu-we-use-deep-networks-to-draw-your-anime-style-portrait-5fbb0ee6b16a) project.

This repository holds the AI model that converts selfies into waifus.

Credit for the model goes to https://github.com/taki0112/UGATIT and https://github.com/t04glovern/UGATIT

The live version of the dapp can be found [here](https://deepwaifu.chronologic.network/).

## 💽 Installation

Run `conda env create -f environment.yml`
and then `conda activate deep_waifu`

(to add new dependencies, change `environment.yml` and then run `conda env update -f environment.yml`)

## 🔧 Model setup

The easiest way is to use the pretrained model provided by https://github.com/taki0112/UGATIT [here](https://drive.google.com/file/d/19xQK2onIy-3S5W5K-XIh85pAg_RNvBVf/view?usp=sharing).

There might be an issue with extracting the file, as mentioned [here](https://github.com/taki0112/UGATIT/issues/87) and [here](https://github.com/taki0112/UGATIT/issues/72). 7-zip on Windows should be able to handle the file.

Once extracted, the model should be placed in the `checkpoint` directory.

## 🚀 Running

Run `python server.py` to start the server

## 🔥 Usage

The server exposes a `POST /selfie2anime` endpoint that accepts Form Data with `file` containing an image and returns a `png` of a waifu.
