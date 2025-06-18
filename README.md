# Efficient LLM-Jailbreaking by Introducing Visual Modality

## 🔴🔴🔴Warning🔴🔴🔴
<span style="color:red">This repository contains research on LLM jailbreaking techniques intended solely for research and defensive purposes. Our work aims to expose security vulnerabilities in existing models to strengthen safety measures in future LLM development. Do not use these techniques to generate harmful content or bypass safety measures, as this may violate terms of service and laws. By using this repository, you agree to use the code responsibly.</span>

## Overview
- Our approach begins by constructing a multimodal large language model (MLLM) through the incorporation of a visual module into the target LLM. Subsequently, we conduct an efficient MLLM-jailbreak to
generate jailbreaking embeddings embJS. Finally, we convert the embJS into text space to facilitate the jailbreaking of the target LLM.
- Compared to direct LLM-jailbreaking, our approach is more efficient, as MLLMs are more vulnerable to jailbreaking than pure LLM.
  
## Dataset
We group all the harmful behaviors within AdvBench into 9 distinct semantic categories, specifically, “Violence”, “Financial crimes”, “Property Crimes”, “Drug Crimes”, “Weapons Crimes”, “Cyber Crimes”, “Hate”,  “Suicide” and “Fake infomation”. At the same time, given that our method involves jailbreaking multimodal large language models, we provide up to thirty images which were retrieved from the Internet using the Google search engine for each class as initialization of the images during the jailbreak process.

## Getting Started

### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/nobody235/LLM-jailbreak.git
cd LLM-jailbreak-main
conda env create -f environment.yml
conda activate MLLM
```

**2. Prepare the pretrained LLM weights**

For llama2, you need to follow the instructions in the minigpt4 repository to set up the model.

For InternLM, you only need to download InternVL2_5-Pretrain-Models-8B.

**3. Prepare LLama-Guard-2**

Download the LLama-Guard-2 at  [Download](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B).  

Then set the LLama-Guard2 path in jailbreak_internlm2.5.py and jailbreak_llama2.py.


If you want to jailbreak other large language models, you can use the suffix generated on LLaMA-2 to jailbreak in a black box manner, or use the MiniGPT-4 method to introduce vision modules to large language models and jailbreak them.



# Launching Demo Locally
First , run [best_init_img.ipynb](best_init_img.ipynb) to get a good initial img for LLM-jb in the next step.

Then , run the following command to jailbreak LLM.
```
python jailbreak_internlm2.5.py  --class_tag S1 --attack_power 128
python jailbreak_llama2.py  --class_tag S1 --attack_power 128
```
  
