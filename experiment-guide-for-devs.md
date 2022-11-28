# Experiment Guide for Devs
Created on 2022-11-28

Hi! :wave: Here's a quick and dirty guide to get to started experimenting with the code we have so far.

## I. Create a virtual machine (VM) on Google cloud
1. Go [here](https://gcp.secure.force.com/GCPEDU?cid=8qQrEkGd0H8GsvikMXIrOhFp89a11IvCa2lptANyWistTURZnoe01KKeoznU836Q/) to redeem your $50 coupon for Google Cloud.
2. Create a project:
    - Go to [Manage Resources page](https://console.cloud.google.com/cloud-resource-manager?walkthrough_id=resource-manager--create-project)
    - Follow [these steps](https://cloud.google.com/resource-manager/docs/creating-managing-projects#creating_a_project).
3. Create a VM.
    - Go to this to set up your VM. `https://console.cloud.google.com/compute/instancesAdd?project=<YOUR_PROJECT_NAME>`
    - GCP will give you several options for creating a VM. Make these changes:
        - Region: `us-west1 (Oregon)`
        - Machine family: `GPU`
        - GPU type: `NVIDIA V100`
        - Machine type: `n1-standard-8`
        - Boot disk: Click `CHANGE`. A dialog box will appear:
            - Operating system: `Deep Learning on Linux`
            - Version: `Debian 10 based Deep Learning VM with M100`
            - Size (GB): `50`
        - Access scopes: `Allow full access to all Cloud APIs`
        - Firewall: `Allow HTTP traffic` and `Allow HTTPS traffic`
    - Click `CREATE`.
4. Boom. Your VM will appear in the [Compute Engine page](https://console.cloud.google.com/compute/instances).

## II. Pull this repo and create an experiment branch just for you
1. SSH into your VM. There's a little `SSH` button to the far right next to your VM name.
2. Once you're in there, clone this repo: `git clone git@github.com:Andyluchina/Citation_Intent_Classification.git`
3. `cd Citation_Intent_Classification`
4. Create a branch for you to make changes: `git checkout -b experiment/<YOUR_NAME>`

## III. Experiment
Here are some key places you can tinker with:
- train.py: check out learning rates, weight decay, regularization

Once you've got a hang of those, you can create more substantial changes for your experiments.

## IV. SUPER IMPORTANT: Delete your VM when not in use!
When you're done experimenting, you'll want to delete your VM. It costs $2 per hour when it's on. We don't know how much it costs when it's off but we'd rather be safe.