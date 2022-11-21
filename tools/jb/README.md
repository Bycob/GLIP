# Auto annotation for DeepDetect

## Setup

- Install requirements.txt (in a virtualenv)
    - CUDA 11.3 required (pytorch above 1.10 not supported)
- Build GLIP
```bash
python setup.py build develop
```
- Download models
```bash
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth
wget https://github.com/Bycob/GLIP/blob/main/configs/pretrain/glip_Swin_T_O365_GoldG.yaml
```
- Run
```bash
python3 auto_annotate.py --model_config MODELS/glip_Swin_T_O365_GoldG.yaml --weights_file MODELS/glip_tiny_model_o365_goldg_cc_sbu.pth images/
```
