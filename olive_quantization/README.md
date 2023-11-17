## 环境配置：

```Plain Text
conda create -n OliVe python=3.8
conda activate OliVe

conda install pytorch=1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

cd ./olive_quantization

pip install -r requirements.txt

pip install ./quant
```



## 适配LLAMA：

配好环境后，在conda环境中更新这些包：

```Plain Text
pip install --upgrade evaluate
pip install datasets -U
pip install --upgrade transformers==4.33
!pip install accelerate==0.20.3
```



## 运行Olive：

```Plain Text
cd olive_quantization/llm
./scripts/run_all.sh
```

run_all.sh中的运行示例：可以按照实验要求手动修改

```Plain Text
CUDA_VISIBLE_DEVICES=1 ./scripts/clm_run.sh LLAMA/llama-7b c4 realnewslike ant-int-flint 4 2 46666 outlier
```

其中：

LLAMA/llama-7b：是存放模型软连接的文件夹，改成opt模型的话：OPT/opt-.7b

c4 realnewslike：是数据集选择，选Wikitext数据集改成：wikitext wikitext-103-raw-v1

4：是bit选择，默认是8bit，这里是4bit

2：batch_size大小

所有脚本参数的设定在 clm_run.sh 文件里