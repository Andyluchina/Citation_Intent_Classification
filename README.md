# Introduction and Motivation

How a scientific publication is cited in another document is important in assessing the former’s impact. Automated analysis of scientific literature must include metrics showing how, not just how many times, a certain publication has been cited - whether as a direct use of a method in that publication, serving as an acknowledgement of previously conducted, or other ways (Cohan, 2019). This type of classification is known as “Citation Intent Classification.”

Understanding how a scientific publication has been used allows for more informed analysis of scientific literature. It is more helpful for researchers to be able to search specifically for publications that are cited as the theoretical basis for newer empirical studies, publications that are cited for the results and benchmarks they describe, or publications that are cited for the datasets they contribute, rather than searching for generic mentions of a publication. Improving models that categorize citations could increase the efficiency and quality of scientific research and literature reviews, and clarify the main findings of existing research (Cohan, 2019; Jurgens, 2018).

We choose to conduct original research exploring how to improve existing models for citation intent classification in scientific publications. We will specifically focus on how to obtain such improvements by integrating BERT and Transformer Neural Networks on top of models that use Bi-LSTM and Multilayer Perceptrons (MLP). Our goal is to determine whether adding improvements such as BERT and Transformer Neural Networks improve the F1 score that was achieved by the Cohan, 2019 paper, which was 67.9.


# Trained model
Our trained model can be found at https://tinyurl.com/citationIntentBestModel

# Before you train: Preparing a Google Compute Engine VM instance
**Optional**: Do this if you have no GPU.
1. Go [here](https://gcp.secure.force.com/GCPEDU?cid=8qQrEkGd0H8GsvikMXIrOhFp89a11IvCa2lptANyWistTURZnoe01KKeoznU836Q/) to redeem your $50 coupon for Google Cloud.
2. Create a project:
    - Go to [Manage Resources page](https://console.cloud.google.com/cloud-resource-manager?walkthrough_id=resource-manager--create-project)
    - Follow [these steps](https://cloud.google.com/resource-manager/docs/creating-managing-projects#creating_a_project).
3. Create a VM.
    - Go to this to set up your VM. `https://console.cloud.google.com/compute/instancesAdd?project=<YOUR_PROJECT_NAME>`
    - GCP will give you several options for creating a VM. Make these changes:
        - Region: `us-west1 (Oregon)`
        - Zone: `us-west1-a`
        - Machine family: `GPU`
        - GPU type: `NVIDIA V100`
        - Machine type: `n1-standard-8`
        - Boot disk: Click `SWITCH IMAGE`. A panel will appear:
            - Operating system: `Deep Learning on Linux`
            - Version: `Debian 10 based Deep Learning VM with M101`
            - Size (GB): `50`
        - Access scopes: `Allow full access to all Cloud APIs`
        - Firewall: `Allow HTTP traffic` and `Allow HTTPS traffic`
    - Click `CREATE`.
4. Your VM will appear in the [Compute Engine page](https://console.cloud.google.com/compute/instances).
5. SSH into your VM. There's a little `SSH` button to the far right next to your VM name.

# Training
ACL ARC dataset size
- train: 1688
- dev: 114
- test: 139

To reinstall Nvidia driver (if needed) run:
`sudo /opt/deeplearning/install-driver.sh`

Enter the following to run the training script. You need a GPU to do this.
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
# Inference
To run inference on the test set of the acl dataset run
```
python3 bert_test.py
```

# Using `screen`
Purpose of this section: To avoid losing progress when you accidentally leave Cloud Shell
```
To resume the session:
screen -r model 

readlink -f bestmodel.npy

y:  Counter({0: 867, 1: 317, 2: 305, 4: 76, 3: 63, 5: 60})
pred:  Counter({0: 1192, 1: 496})
```

screen guides
```
screen -S model 
///creating a session called model
Once you have the model running with training script
Press Ctrl+a+d to detach from the session

if you want to resume the session because of loss of connection, run
screen -r model 
////this will resume the session that you were in
```

