
# %%
import os
import joblib
from IPython.display import display, HTML

# Define the path to the data
base_path = os.path.dirname("//nmshare.corp.nm.org/nmdata/imaging-mozzi/Projects/Active/incidentals/LanguageModel/Phase1Phase2DemoScripts/datasets/final_df_modeling.gz")
data_path = os.path.abspath(os.path.join(base_path, "final_df_modeling.gz"))


# Import data
modeling_df = joblib.load(data_path)

display(HTML(modeling_df.head(4).to_html()))
# %%
display(HTML(modeling_df.head(9).to_html()))
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
#Why is there a b' that is added to the beginning of each entry in the impression column? and why are they adding the new lines at the end?
# %%
from sklearn.model_selection import train_test_split

# Split into train and test data
train, test = train_test_split(modeling_df, test_size=0.2, random_state=7867)
train = train.reset_index()
test = test.reset_index()
# %%
from datasets import Dataset, DatasetDict

# Import the data into a dataset
train_dataset = Dataset.from_pandas(train["impression"].to_frame())
test_dataset = Dataset.from_pandas(test["impression"].to_frame())
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset
# %%
from transformers import AutoTokenizer

# Specify the model checkpoint for tokenizing and get tokenizer
tokenizer = AutoTokenizer.from_pretrained(
        "StanfordAIMI/RadBERT",
        use_fast=True,
        padding_side="left",
    )

#Stanford model has no maximum length so we need to define it! (RoBERTa has 512)
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["impression"], truncation=True, padding=True),
    batched=True,
    num_proc=1,
    remove_columns=["impression"],
)

# %%
def group_texts(examples):
    # Sample chunked into size `block_size`
    block_size = 128

    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder. We could add padding if the model supported it rather than dropping it.
    # This represents the maximum length based on the block size
    # You can customize this part to your needs.
    max_length = (total_length // block_size) * block_size
    result = {k: [t[i : i + block_size] for i in range(0, max_length, block_size)] for k, t in concatenated_examples.items()}
    result["labels"] = result["input_ids"].copy()

    return result


# Group the text into chunks to get "sentence-like" data structure
lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=1,
)
# %%
#1 is the padding token, appears a lot
set_sample = lm_dataset["train"].shuffle(seed=44).select(range(1000))
#Try replacing 4 with 7,16,166
tokenizer.decode(set_sample[7]['input_ids'])
# %%
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,  mlm_probability=0.15)

# %%
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments

# Define the model
model = AutoModelForMaskedLM.from_pretrained("StanfordAIMI/RadBERT")

# Define the training parameters and 🤗 Trainer
training_args = TrainingArguments(
    output_dir="/path/to/results/phase02/demo",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    fp16=True,
    save_steps=2,
    save_total_limit=2,
    evaluation_strategy="epoch",
    seed=1,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)


# %%
trainer.train()
# %%
