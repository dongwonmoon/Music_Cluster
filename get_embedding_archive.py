import os
import random
import json
import warnings
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# 사전 학습된 feature extractor와 모델 로드
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "jalal-elzein/distilhubert-finetuned-gtzan"
)
model = AutoModelForAudioClassification.from_pretrained(
    "jalal-elzein/distilhubert-finetuned-gtzan"
)
model.eval()  # 모델을 평가 모드로 전환


def get_embedding(source_dir, sample_frac=0.1):
    """
    source_dir의 모든 오디오 파일을 처리하여 임베딩을 추출합니다.
    오디오 파일의 샘플링 레이트가 예상 값과 다를 경우 경고 표시합니다.
    """
    records = []
    for composer in tqdm(os.listdir(source_dir), desc="작곡가"):
        composer_path = os.path.join(source_dir, composer)
        if not os.path.isdir(composer_path):
            continue
        for music_file in tqdm(
            os.listdir(composer_path), desc="음악 파일", leave=False
        ):
            audio_path = os.path.join(composer_path, music_file)
            try:
                audio_array, sampling_rate = sf.read(audio_path)

                total_length = len(audio_array)
                segment_length = int(sample_frac * total_length)

                max_start = total_length - segment_length
                random_start = random.randint(0, max_start)
                random_end = random_start + segment_length
                audio_segment = audio_array[random_start:random_end]

                processed = feature_extractor(
                    audio_segment,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=False,
                )

                outputs = model.hubert(**processed)
                embedding = outputs.last_hidden_state.mean(dim=1)
                embedding_list = embedding.squeeze().detach().cpu().numpy().tolist()

                records.append(
                    {
                        "composer": composer,
                        "music": music_file,
                        "embedding": embedding_list,
                    }
                )

            except Exception as e:
                print("오류 발생, 건너뜀:", music_file, ":", e)

    return records


if __name__ == "__main__":
    source_directory = "./data/archive_wav"
    records = get_embedding(source_directory)
    print("총 추출된 임베딩 개수:", len(records))

    # DataFrame 생성 및 임베딩 리스트를 JSON 문자열로 변환 후 CSV 저장
    df = pd.DataFrame(records)
    df["embedding"] = df["embedding"].apply(lambda emb: json.dumps(emb))
    output_csv = "./data/embeddings_classic.csv"
    df.to_csv(output_csv, index=False)
    print("임베딩과 메타 정보를 CSV 파일로 저장했습니다:", output_csv)
