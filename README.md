# embrain

embrain 웹서비스 repository입니다.

## Opencoding

### Versions

- Python version: 3.8.13
- CUDA version: 11.7
- Cudnn version: 8500

----------

### Training Data structure
  |prompt|question|input|completion|
  |:---:|:---:|:---:|:---:|
  |"다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘."|귀하께서 알고 계시는...|친근하고 맛있는 과자|<NULL, 친숙하다> <NULL, 맛있다>| <br>

  * 파일 형태: json 파일

----------

### How to install
  ```sh
git clone https://github.com/skkuembrain/embrain
cd opencoding
pip install -r requirements.txt
```

----------

### Finetuning

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
- answer_log.txt / answer_log.xlsx : 전체 모델 테스트 로그
- error_log.txt / error_log.xlsx : 틀린 생성 값들에 대한 테스트 로그

----------

### Inference

Arguments in `inference.py`:
  |Parameter name|Description|Default|Options|
  |:---:|:---:|:---:|:---:|
  |`--model`|사용한 베이스 모델|X|'kogpt2', 'polyglot', 'trinity', 'kogpt'|
  |`--model_dir`|테스트 할 모델 주소|X|string|
  |`--save_dir`|테스트 결과 저장 위치|"./"|string|
  |`--test_file`|테스트 할 엑셀 파일|X|엑셀 파일|

  * 테스트 엑셀은 '질문 내용'과 '응답' 항목 필요

예시:
  ```sh
python inference.py --model=kogpt2 --model_dir="./models/kogpt2/checkpoint-50000" --save_dir="./test_result" --test_file="./test_excel.xlsx"
```

