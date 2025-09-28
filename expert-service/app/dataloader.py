import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import warnings

# Suppress the specific UserWarning from torch.tensor on a tensor
# We are handling this correctly with torch.stack now, but this is good practice.
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor.*")

# --- 1. The Dataset Class ---
class AGNewsDataset(Dataset):
    """
    A PyTorch Dataset class to handle the AG News data.
    It processes the raw text iterator into a list of tensors in the constructor.
    """
    def __init__(self, data_iterator, vocab, tokenizer):
        self.data = []
        self.vocab = vocab
        self.tokenizer = tokenizer

        # This loop consumes the iterator and stores the processed data in self.data
        for label, text in data_iterator:
            tokens = self.tokenizer(text)
            indices = self.vocab(tokens)
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            label_tensor = torch.tensor(label - 1, dtype=torch.long) # Adjust label from 1-4 to 0-3
            self.data.append((label_tensor, indices_tensor))

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the processed sample at a given index."""
        return self.data[idx]

# --- 2. The Main Function to Get Dataloaders and Vocab ---
def get_dataloaders_and_vocab(batch_size=64):
    """
    Orchestrates the entire data loading process.
    Handles iterator exhaustion correctly by creating fresh iterators for each step.

    Args:
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple: A tuple containing (train_dataloader, test_dataloader, vocab)
    """
    print("--- Starting Data Loading Process ---")

    # --- Step A: Setup Tokenizer ---
    tokenizer = get_tokenizer('basic_english')

    # --- Step B: Build Vocabulary ---
    # We create a fresh iterator here specifically for building the vocabulary.
    # This iterator will be exhausted after this step.
    print("Building vocabulary...")
    train_iter_for_vocab, _ = AG_NEWS(split=('train', 'test'))

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter_for_vocab), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    print(f"Vocabulary built. Size: {len(vocab)}")

    # --- Step C: Instantiate Datasets ---
    # We create a SECOND set of fresh iterators to pass to our Dataset class.
    print("Processing data and creating Dataset objects...")
    train_iter_for_dataset, test_iter_for_dataset = AG_NEWS(split=('train', 'test'))

    train_dataset = AGNewsDataset(train_iter_for_dataset, vocab, tokenizer)
    test_dataset = AGNewsDataset(test_iter_for_dataset, vocab, tokenizer)
    print("Dataset objects created.")

    # --- Step D: Define the Collate Function ---
    # This function defines how to combine a list of samples into a single batch.
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(_label)
            text_list.append(_text)

        # Use torch.stack for labels, as it's the correct way to combine a list of tensors.
        labels_tensor = torch.stack(label_list)

        # Use pad_sequence for text to handle variable lengths.
        texts_tensor = pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])

        return texts_tensor, labels_tensor # Note: Returning (text, label) is more conventional

    # --- Step E: Create DataLoaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    print("--- Data Loading Process Complete ---")

    return train_dataloader, test_dataloader, vocab

def get_tokenizer_and_vocab():
    """
    A lightweight function that only builds and returns the tokenizer and vocab.
    """
    tokenizer = get_tokenizer('basic_english')
    train_iter = AG_NEWS(split='train')
    
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return tokenizer, vocab

# --- 3. Self-Execution Block for Testing ---
if __name__ == '__main__':
    """
    This block runs only when the script is executed directly.
    It's a good way to test that the file is working correctly on its own.
    """
    print("Testing dataloader.py directly...")

    train_dl, test_dl, vocab_obj = get_dataloaders_and_vocab(batch_size=8)

    print(f"\nVocabulary size: {len(vocab_obj)}")

    # Get a single batch from the training dataloader
    text_batch, labels_batch = next(iter(train_dl))

    print("\n--- Testing a single batch ---")
    print(f"Text batch shape: {text_batch.shape}")   # Should be [8, seq_len]
    print(f"Labels batch shape: {labels_batch.shape}") # Should be [8]

    print("\nFirst text tensor in batch:\n", text_batch[0])
    print("\nFirst label in batch:", labels_batch[0])
    print("\ndataloader.py test successful!")

