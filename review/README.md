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
  |강의하시는 선생님이 재미있고, 덕분에 시험 성적도 올랐다.|강사가 재미있고, 성적이 올랐다.|• 강사가 재미있음<br>• 성적 상승|긍정|<br>
### for only one task: (ex) keyphrase)
  |input|completion|
  |:---:|:---:|
  |강의하시는 선생님이 재미있고, 덕분에 시험 성적도 올랐다.|• 강사가 재미있음<br>• 성적 상승|<br>

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
  |`--model`|사용할 베이스 모델|X|'kogpt', 'trinity'|
  |`--epochs`|학습 에포크|32|int > 0|
  |`--batch_size`|학습 배치 사이즈|4|int > 0|
  |`--l_rate`|러닝 레이트|3e-05|3e-04 ~ 5e-05|
  |`--save_dir`|모델 저장 주소(./model로 설정시 ./model/model_id/epoch_lr에 저장)|'./model'|string|
  |`--dataset`|학습할 때 사용하는 데이터셋|X|xlsx 파일|
  |`--mode`|학습할 task 모드|3|0(요약), 1(핵심구문추출), 2(감성분석), 3(전체)|

예시:
  ```sh
python3 main.py --model=kogpt --epochs=32 --batch_size=8 --l_rate=3e-05 --save_dir="./model" --dataset="./dataset/dataset.xlsx --mode=3"
```

----------

## Inference

Arguments in `inference.py`:
  |Parameter name|Description|Default|Options|
  |:---:|:---:|:---:|:---:|
  |`--model`|사용한 베이스 모델|X|'kogpt', 'trinity'|
  |`--model_dir`|테스트 할 모델 주소|X|string|
  |`--save_dir`|테스트 결과 저장 위치|"./result"|string|
  |`--test_file`|테스트 할 엑셀 파일|X|엑셀 파일|
  |`--mode`|테스트할 task 모드|3|0(요약), 1(핵심구문추출), 2(감성분석), 3(전체)|

  * 테스트 엑셀이 '질문 내용'만 포함한 경우 생성값만 엑셀로 출력(추론)
  * 테스트 엑셀이 '질문 내용'과 '정답'을 포함한 경우 생성값 및 점수를 엑셀로 출력(테스트)

예시:
  ```sh
python3 inference.py --model=kogpt --model_dir="./model/kogpt/1e-05/checkpoint-50000" --save_dir="./result" --test_file="./dataset/test_data.xlsx --mode=3"
```

