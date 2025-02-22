import io
import soundfile as sf
import pandas as pd
from tqdm import tqdm

from datasets import DownloadManager
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import json

# 사전 학습된 feature extractor와 모델을 불러옵니다.
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "jalal-elzein/distilhubert-finetuned-gtzan"
)
model = AutoModelForAudioClassification.from_pretrained(
    "jalal-elzein/distilhubert-finetuned-gtzan"
)

# 데이터셋의 tar.gz 아카이브 URL
url = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"


def get_embedding(archive_url):
    """
    주어진 URL의 tar.gz 아카이브를 다운로드한 후, .wav 파일들을 추출하여
    feature extractor로 전처리하고, 모델에 통과시켜 임베딩(모델 출력)을 얻습니다.
    각 파일의 filename과 임베딩 정보를 함께 저장합니다.

    인자:
        archive_url (str): tar.gz 아카이브 URL

    반환값:
        records (list of dict): 각 오디오 파일에 대한 filename과 임베딩(목록)의 딕셔너리 리스트
    """
    dl_manager = DownloadManager()
    archive_path = dl_manager.download(archive_url)

    records = []
    for filename, file_obj in tqdm(
        dl_manager.iter_archive(archive_path), desc="Processing archive"
    ):
        # .wav 파일만 처리합니다.
        if not filename.lower().endswith(".wav"):
            continue

        try:
            file_bytes = file_obj.read()
            audio_file = io.BytesIO(file_bytes)

            audio_array, sampling_rate = sf.read(audio_file)

            processed = feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=False,
            )

            outputs = model.hubert(**processed)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embedding_list = embedding.squeeze().detach().cpu().numpy().tolist()

            records.append({"filename": filename, "embedding": embedding_list})

        except Exception as e:
            print("오류로 건너뜀", filename, ":", e)

    return records


if __name__ == "__main__":
    records = get_embedding(url)
    print("총 추출된 임베딩 개수:", len(records))

    df = pd.DataFrame(records)
    df["embedding"] = df["embedding"].apply(lambda emb: json.dumps(emb))

    output_csv = "./data/embeddings_with_metadata.csv"
    df.to_csv(output_csv, index=False)
    print("임베딩과 메타 정보를 CSV 파일로 저장했습니다:", output_csv)
