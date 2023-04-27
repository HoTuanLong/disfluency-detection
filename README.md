# Vietnamese Disfluency Detection
## Overview
This repository contains code for a Vietnamese disfluency detection system, which is a reimplementation of the method proposed in the paper "Disfluency Detection for Vietnamese" by Dao et al. The system is implemented using the HuggingFace framework.

## Usage
To use the Vietnamese disfluency detection system, follow these steps:

1. Install the required packages by running the following command:
```
pip install -r requirements.txt
```

2. Clone this repo
```
git clone https://github.com/your_username/vietnamese-disfluency-detection.git
cd vietnamese-disfluency-detection
```

3. Training and Testing
```
python source/main/train.py
python source/main/test.py
```

## Credits
This implementation is based on the paper "Disfluency Detection for Vietnamese" by Dao et al. [1].

[1] Dao, M., Truong, T.H., & Nguyen, D.Q. (2022). Disfluency Detection for Vietnamese. In Proceedings of the Eighth Workshop on Noisy User-generated Text (W-NUT 2022), pp. 194-200.

The pre-trained model used in this implementation is hosted on the Hugging Face model hub by the user longht, at: https://huggingface.co/spaces/longht/vietnamese-disfluency-detection.

