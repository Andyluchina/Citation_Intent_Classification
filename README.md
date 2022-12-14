Introduction and Motivation

How a scientific publication is cited in another document is important in assessing the former’s impact. Automated analysis of scientific literature must include metrics showing how, not just how many times, a certain publication has been cited - whether as a direct use of a method in that publication, serving as an acknowledgement of previously conducted, or other ways (Cohan, 2019). This type of classification is known as “Citation Intent Classification.”

Understanding how a scientific publication has been used allows for more informed analysis of scientific literature. It is more helpful for researchers to be able to search specifically for publications that are cited as the theoretical basis for newer empirical studies, publications that are cited for the results and benchmarks they describe, or publications that are cited for the datasets they contribute, rather than searching for generic mentions of a publication. Improving models that categorize citations could increase the efficiency and quality of scientific research and literature reviews, and clarify the main findings of existing research (Cohan, 2019; Jurgens, 2018).

We choose to conduct original research exploring how to improve existing models for citation intent classification in scientific publications. We will specifically focus on how to obtain such improvements by integrating BERT and Transformer Neural Networks on top of models that use Bi-LSTM and Multilayer Perceptrons (MLP). Our goal is to determine whether adding improvements such as BERT and Transformer Neural Networks improve the F1 score that was achieved by the Cohan, 2019 paper, which was 67.9.


ACL ARC dataset size
train: 1688 \
dev: 114 \
test: 139 \
To reinstall Nvidia driver (if needed) run:
`sudo /opt/deeplearning/install-driver.sh`

```
nvidia-smi
git clone https://github.com/Andyluchina/Citation_Intent_Classification
cd Citation_Intent_Classification
pip install -r requirements.txt
tar -xvf acl-arc.tar.gz 
tar -xvf scicite.tar.gz 
screen -S model
python3 train.py

```
To resume the session:
screen -r model 


readlink -f bestmodel.npy

y:  Counter({0: 867, 1: 317, 2: 305, 4: 76, 3: 63, 5: 60})
pred:  Counter({0: 1192, 1: 496})
```

screen guides

screen -S model 
///creating a session called model
Once you have the model running with training script
Press Ctrl+a+d to detach from the session

if you want to resume the session because of loss of connection, run
screen -r model 
////this will resume the session that you were in