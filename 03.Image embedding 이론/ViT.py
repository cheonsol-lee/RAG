import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
dataset_train = load_dataset("keremberke/shoe-classification",'full', split='train')

#사전학습 Vision 트랜스포머 불러오기
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.neighbors import NearestNeighbors
import torch

# import model - https://huggingface.co/google/vit-base-patch16-224-in21k
model_id = 'google/vit-base-patch16-224-in21k'

feature_extractor_vanilla = ViTImageProcessor.from_pretrained(model_id)

def extract_embeddings(dataset, model, feature_extractor):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for item in dataset:
            inputs = feature_extractor(images=item['image'], return_tensors="pt")
            outputs = model(**inputs)
            embeddings.append(outputs.logits.squeeze().numpy())
    return embeddings


def retrieve_images(index, nn_model, dataset, embeddings):
    index = int(index)
    distances, indices = nn_model.kneighbors([embeddings[index]])

    indexed_distances = [(int(i), dist) for i, dist in zip(indices[0], distances[0]) if i != index]

    indexed_distances.sort(key=lambda x: x[1])

    retrieved_images = [dataset[idx]["image"] for idx, _ in indexed_distances]
    return retrieved_images

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        img = dataset["train"][i]["image"]
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

#Fine-Tuning
dataset_test = load_dataset("keremberke/shoe-classification",'full', split='validation')


def preprocess(batch):
    inputs = feature_extractor_vanilla(
        batch['image'],
        return_tensors='pt'
    )
    inputs['labels'] = batch['labels']
    return inputs

train_prepared = dataset_train.with_transform(preprocess)
test_prepared = dataset_test.with_transform(preprocess)

# 배치 텐서별 텐서와 레이블을 올바르게 정렬
def collate_batch(batch):
    pixel_vals = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {'pixel_values': torch.stack(pixel_vals), 'labels': torch.tensor(labels)}

import numpy as np
from datasets import load_metric

accuracy_metric = load_metric("accuracy")

def evaluate_model_performance(outputs):
    predicted_labels = np.argmax(outputs.predictions, axis=1)
    true_labels = outputs.label_ids
    return accuracy_metric.compute(predictions=predicted_labels, references=true_labels)


num_labels_in_dataset = len(set(dataset_train['labels']))
label_names = dataset_train.features['labels'].names

# ViTForImageClassification 바닐라 모델은 출력 레이블 2개 였지만, 
# 새롭게 파인튜닝 시킬 레이블 갯수는 3개
model = ViTForImageClassification.from_pretrained(
    model_id,
    num_labels=num_labels_in_dataset
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import transformers
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./../../shoe",
  per_device_train_batch_size=128,
  evaluation_strategy="steps",
  num_train_epochs=10,
  save_steps=20,
  eval_steps=20,
  logging_steps=20,
  learning_rate=0.0002,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  #load_best_model_at_end=True,
)

model.to(device)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_batch,
    compute_metrics=evaluate_model_performance,
    train_dataset=train_prepared,
    eval_dataset=test_prepared,
    tokenizer=feature_extractor_vanilla,
)

training_outcome = trainer.train()
trainer.save_model()

training_metrics = training_outcome.metrics
trainer.log_metrics("training", training_metrics)
trainer.save_metrics("training", training_metrics)

trainer.save_state()


evaluation_metrics = trainer.evaluate(test_prepared)

trainer.log_metrics("evaluation", evaluation_metrics)
trainer.save_metrics("evaluation", evaluation_metrics)