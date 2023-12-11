# Review

## Versions

- Python version: 3.8.13
- CUDA version: 11.7
- Cudnn version: 8500

----------

## Training Data structure
### for all task: 
  |input|summary_completion|keyphrase_completion|sentiment-analysis_completion|
  |:---:|:---:|:---:|:---:|
  |강의하시는 선생님이 재미있고, 덕분에 시험 성적도 올랐다.|강사가 재미있고, 성적이 올랐다.|* 강사가 재미있음\n* 성적 상승|긍정|<br>
### for only one task: (ex) keyphrase)
  |input|completion|
  |:---:|:---:|
  |강의하시는 선생님이 재미있고, 덕분에 시험 성적도 올랐다.|* 강사가 재미있음\n* 성적 상승|<br>

  * 파일 형태: xlsx 파일
  * prompt는 xlsx -> json 변환 코드에 의해 task에 맞게 자동 삽입

----------

## How to install
  ```sh
git clone https://github.com/skkuembrain/embrain
cd review
pip install -r requirements.txt
```

----------

## Finetuning

Arguments in `main.py`:
  |Parameter name|Description|Default|Options|
  |:---:|:---:|:---:|:---:|
  |`--model`|사용할 베이스 모델|X|'kogpt2', 'polyglot', 'trinity', 'kogpt'|
  |`--epochs`|학습 에포크|20|int > 0|
  |`--batch_size`|학습 배치 사이즈|8|int > 0|
  |`--save_step`|모델 저장 스탭|500|int > 0|
  |`--save_dir`|모델 저장 주소|'./model'|string|
  |`--dataset`|학습할 때 사용하는 데이터셋|X|json 파일|

예시:
  ```sh
python main.py --model=kogpt2 --epochs=50 --batch_size=8 --save_step=500 --save_dir="./models/kogpt2" --dataset="./Datasets/dataset.json"
```

테스트 로그:
- 저장 위치: {save_dir}/logs
- loss_log.txt / loss_log(epoch).png : 학습 loss 로그
- answer_log.txt / answer_log.xlsx : 전체 테스트 로그
- error_log.txt / error_log.xlsx : 틀린 생성 값들에 대한 테스트 로그

----------

## Inference

Arguments in `inference.py`:
  |Parameter name|Description|Default|Options|
  |:---:|:---:|:---:|:---:|
  |`--model`|사용한 베이스 모델|X|'kogpt2', 'polyglot', 'trinity', 'kogpt'|
  |`--model_dir`|테스트 할 모델 주소|X|string|
  |`--save_dir`|테스트 결과 저장 위치|"./result"|string|
  |`--test_file`|테스트 할 엑셀 파일|X|엑셀 파일|

  * 테스트 엑셀은 '질문 내용'과 '응답' 항목 필요

예시:
  ```sh
python inference.py --model=kogpt2 --model_dir="./models/kogpt2/checkpoint-50000" --save_dir="./test_result" --test_file="./test_excel.xlsx"
```

