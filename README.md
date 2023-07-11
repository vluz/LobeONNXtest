# Lobe ONNX test
### A Python cli app to test both Lobe and ONNX runtime

Created a subset of the Asirra dataset, removed bad images, text, people, blurry, and other troubles.      
Dataset is composed of two categories, Cat and Dog with 150 images on each.      
It's ideal for this kind of a test as it trains in seconds and provides good prediction.       

https://www.kaggle.com/datasets/vicluz/mini-asirra-150

<hr>

Trained with Lobe beta and exported as ONNX.

https://huggingface.co/vluz/MiniAsirraONNX

<hr>

Open a command prompt and `cd` to a new directory of your choosing:

(optional; recommended) Create a virtual environment with:
```
python -m venv "venv"
venv\Scripts\activate
```

To install do:
```
git clone https://github.com/vluz/LobeONNXtest.git
cd LobeONNXtest
pip install -r requirements.txt
```

Dowload the model and configuration file from here:      
https://huggingface.co/vluz/MiniAsirraONNX      

Two files to download, `model.onnx` and `signature.json`      

Put both files it in the same directory as `onnxtest.py`

To run do:<br>
`python onnxtest.py test1.jpg`      
or      
`python onnxtest.py test2.jpg`      
You can also use any image containing either a dog or a cat.

<hr>

Output:       

```
{
    "predictions": [
        {
            "label": "Cat",
            "confidence": 1.0
        },
        {
            "label": "Dog",
            "confidence": 5.928525402313069e-34
        }
    ]
}
```

<hr>

Do not use for production, untested.

<br>
