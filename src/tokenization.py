import pandas as pd
from transformers import BertTokenizer
from pathlib import Path


def tokenize_and_save(df, tokenizer, output_dir, file_suffix, max_length=512):
    print(f"Tokenizing {file_suffix} data...")
    tokenized_data = tokenizer(
        df['input_text'].tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Convert to DataFrame
    tokens_df = pd.DataFrame(tokenized_data['input_ids'].numpy())
    masks_df = pd.DataFrame(tokenized_data['attention_mask'].numpy())

    # Saving the tokenized data
    token_output_file = Path(output_dir) / f"{file_suffix}_tokenized_input_ids.csv"
    mask_output_file = Path(output_dir) / f"{file_suffix}_tokenized_attention_masks.csv"

    tokens_df.to_csv(token_output_file, index=False)
    masks_df.to_csv(mask_output_file, index=False)

    print(f"Saved the tokenized {file_suffix} data to {output_dir}")


def tokenize_data(input_filepath, output_dir, tokenizer, max_length=512, is_split=False):
    tokenizer = BertTokenizer.from_pretrained(tokenizer)

    if is_split:
        # Tokenize train data
        train_df = pd.read_csv(Path(input_filepath) / 'train_prompts.csv')
        tokenize_and_save(train_df, tokenizer, output_dir, 'train', max_length)

        # Tokenize validation data
        validation_df = pd.read_csv(Path(input_filepath) / 'validation_prompts.csv')
        tokenize_and_save(validation_df, tokenizer, output_dir, 'validation', max_length)
    else:
        # Tokenize full data
        full_df = pd.read_csv(Path(input_filepath) / 'prompts.csv')
        tokenize_and_save(full_df, tokenizer, output_dir, '', max_length)


if __name__ == "__main__":
    # IMDB Data Tokenization (which has split data)
    imdb_input_dir = '../data/processed/imdb'  # Update this path if necessary
    imdb_output_dir = '../data/tokenized/imdb'  # Update this path if necessary
    tokenize_data(imdb_input_dir, imdb_output_dir, 'bert-base-uncased', is_split=True)

    # Financial Data Tokenization (no split data, used only for testing)
    financial_input_filepath = '../data/processed/financial'  # Update this path if necessary
    financial_output_dir = '../data/tokenized/financial'  # Update this path if necessary
    tokenize_data(financial_input_filepath, financial_output_dir, 'bert-base-uncased', is_split=False)
