# Video to ARKit Blendshapes
Use `mediapipe` to detect changes in facial expressions in the video and output a JSON file of ARKit Blendshapes.

## Install
### Conda env create
```
conda create --name video2face
conda activate video2face
```
### Dependent installation
```
pip install -r requirements.txt
```
### Run
```python
python run.py --input ./videos/ --output ./outputs/
```