## Quick-Start

If you wish to use our dataset, you must first obtain permission from [https://zenodo.org/records/13922703](https://zenodo.org/records/13922703). Our dataset is shared under a confidentiality agreement.

> In accordance with the recently enacted Personal Information Protection Law and the Data Security Assessment Measures for Cross-Border Data Transfers in China, we currently provide access to the dataset only to Chinese institutions (universities, research institutes, and companies). To proceed, please send us your name and institutional details. We will respond with the relevant confidentiality agreement. Only after signing the agreement will we grant access to the dataset.


```bash
git clone https://github.com/lyingCS/LCU

wget https://zenodo.org/record/13922703/files/LCU.zip
wget https://zenodo.org/record/13922581/files/KuaiComt.zip
mkdir LCU/rec_datasets/KuaiComt
mkdir LCU/rec_datasets/WM_KuaiComt
unzip -d LCU/rec_datasets/KuaiComt LCU.zip
unzip -d LCU/rec_datasets/KuaiComt KuaiComt.zip

cd LCU/src
bash run.sh
```
