# CommonLit Readability

NLP model for scoring text passages in English according to [Kaggle competition](https://www.kaggle.com/competitions/commonlitreadabilityprize).

Here we are using [BERT](https://huggingface.co/distilbert-base-uncased) for encoding input text and transfer learning with simple regression model to train on [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize) dataset. 

## How to train the model
1. Clone the repository
```bash
git clone https://github.com/rosklyar/commonlit-readability.git
```
2. Move to the repository directory
```bash
cd commonlit-readability
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run training script
```bash
python train.py
```
5. Run tensorboard to monitor training process
```bash
tensorboard --logdir=checkpoints/commonlit_nn
```
Each n epochs model will be saved to `checkpoints/commonlit_nn` directory with RMSE metric suffix `_RMSE={metric_value}.pth`.

## How to run docker image and make predictions
1. Copy trained model to 'app\commonlit-model.pth'
2. Create docker image
```bash
docker build -t commonlit-readability .
```
3. Run docker image
```bash
docker run -d --name commonlit-readability-container -p 80:80 commonlit-readability
```
4. Make prediction
```bash
curl -X POST http://127.0.0.1/score -H "Content-Type: application/json" -d "{\"text\":\"This is simple test text\"}"
```

## Check deployed model on Google Cloud
```bash
curl -X POST https://commonlit-web-cqeyyrdvka-ew.a.run.app/score -H "Content-Type: application/json" -d "{\"text\":\"This is simple test text\"}"
```