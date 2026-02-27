
# Manual installation and operation
## 1.Install and few-shot
You can run the code according to the following method. The code below is for performing the few-shot step

step1: 
```
conda create -n instruct_sql -c conda-forge python=3.10 openjdk=17 nccl  
conda activate instruct_sql
``` 
step2:  
To set up the environment, you should download the stanford-cornlp（http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip） and unzip it to the folder ./model/third_party. Next, you need to launch the coreNLP server:  
```
cd model/third_party/stanford-corenlp-full-2018-10-05  
nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &  
cd ../../../
```
step3:  
```
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121  
pip install -r instruct_requirements.txt
``` 
step4:  
```
cd src  
export OPENROUTER_API_KEY=sk-xxxxxxx  # Fill in your api_key here  
python nltk_downloader.py  
sh run_for_bird.sh  test   #If you want to test the test set, please change "dev" to "test" in the sh file  
cd ..
#It takes 1-2 hours to run here
``` 

## 2.Preprocess and Main
```
sh run.sh test  # If you want to test the test set, please change "dev" to "test" in the commend
```               
