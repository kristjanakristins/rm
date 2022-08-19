# %%
import os
import joblib
from IPython.display import display, HTML
import wandb

os.chdir('/workspace')
# Define the path to the data
base_path = os.path.dirname("./data/datasets/final_df_modeling.gz")
data_path = os.path.abspath(os.path.join(base_path, "final_df_modeling.gz"))


# Import data
modeling_df = joblib.load(data_path)

display(HTML(modeling_df.head(4).to_html()))
# %%

def keyword_split(x, keywords, return_idx: int=2):
    """
    Extract portion of string given a list of possible delimiters (keywords) via partition method
    """
    for keyword in keywords:
        if x.partition(keyword)[2] !='':
            return x.partition(keyword)[return_idx]
    return x

def preprocess_note(note):
    """
    Get the impression from the note, remove doctor signature, and lowercase
    """
    impression_keywords = [
            "impression:",
            "conclusion(s):",
            "conclusions:",
            "conclusion:",
            "finding:",
            "findings:",
    ]
    signature_keywords = [
        "&#x20",
        "final report attending radiologist:",
    ]
    impressions = keyword_split(str(note).lower(), impression_keywords)
    impressions = keyword_split(impressions, signature_keywords, return_idx=0)
    return impressions

# Preprocess the note
modeling_df["impression"] = modeling_df["note"].apply(preprocess_note)
modeling_df = modeling_df[modeling_df["impression"].notnull()]
modeling_df["impression"] = modeling_df["impression"].apply(lambda x: str(x.encode('utf-8')) +"\n"+"\n")
# %%
from sklearn import preprocessing

# Encode the Lung, Adrenal, and No Finding into integer labels
le = preprocessing.LabelEncoder()
le.fit(modeling_df["selected_finding"])
modeling_df["int_labels"] = le.transform(modeling_df["selected_finding"])
# %%
from sklearn.model_selection import train_test_split

# Split the data into train and test
train_df, test_df = train_test_split(modeling_df, test_size=0.3, stratify=modeling_df["selected_finding"], random_state=37)
train_note = list(train_df["impression"])
train_label = list(train_df["int_labels"])
test_note = list(test_df["impression"])
test_label = list(test_df["int_labels"])
# %%
from transformers import AutoTokenizer

# Define the tokenizer (from a pre-trained checkpoint) and tokenize the notes USING THE DEFAULT PADDING SIDE
tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/covid-radbert", use_fast=True)
train_encodings = tokenizer(train_note, truncation=True, padding=True, max_length = 512)
val_encodings = tokenizer(test_note, truncation=True, padding=True, max_length = 512)
# %%
import torch

class Reports_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list) -> None:
        self.encodings = encodings
        self.labels = labels
        return

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)

# Define the trainign dataset with tokenized notes and labels
train_dataset = Reports_Dataset(train_encodings, train_label)
val_dataset = Reports_Dataset(val_encodings, test_label)
# %%
from transformers import AutoModelForSequenceClassification

# TODO: point to the pretrained model trained as part of the pretraining process
# Here, we are using a pretrained checkpoint trained on thousands of reports (vs the pretrained model wieghts generated via the notebook ``demo_pretrain``)
# To use the only directly trained by the notebook, use "/path/to/results/phase02/demo/checkpoint-4"

#model_pretrained_path = "/path/to/results/phase02/demo/checkpoint-14500"

# Fine-tune the model from the pre-trained checkpoint
model = AutoModelForSequenceClassification.from_pretrained("StanfordAIMI/covid-radbert", num_labels=3)

# %%
from transformers import Trainer, TrainingArguments

# Define the training parameters and 🤗 Trainer
training_args = TrainingArguments(
                    output_dir="/path/to/results/phase02/demo/findings",    # output directory
                    num_train_epochs=2,                                    # total number of training epochs
                    per_device_train_batch_size=16,                         # batch size per device during training
                    per_device_eval_batch_size=8,                           # batch size per device during evaluation
                    warmup_steps=100,                                       # number of warmup steps for learning rate scheduler
                    weight_decay=0.015,                                     # strength of weight decay
                    fp16=True,                                              # mixed precision training
                    do_predict=True,                                        # run predictions on test set
                    load_best_model_at_end=True,                            # load best model at end so we can run confusion matrix
                    logging_steps=2,                                        # remaining args are related to logging
                    save_total_limit=2,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    report_to="none",
)
trainer = Trainer(
                    model=model,                                            # the instantiated 🤗 Transformers model to be trained
                    args=training_args,                                     # training arguments, defined above
                    train_dataset=train_dataset,                            # training dataset
                    eval_dataset=val_dataset,                               # test (evaluation) dataset: save and eval strategy to match
)
# %%
trainer.train()

# %%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Perform confusion matrix and print the results
y_pred = trainer.predict(val_dataset)
y_pred = np.argmax(y_pred.predictions, axis=1)
report = classification_report(test_label, y_pred)
matrix = confusion_matrix(test_label, y_pred)
print(report)
print(matrix)
# %%
specificity1 = matrix[0,0]/(matrix[0,0]+matrix[0,1]+matrix[0,2])


os.makedirs("./models")
model.save_pretrained("./models")

# %%
