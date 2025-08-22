import logging
from torch.utils.data import Dataset, DataLoader
from config import config

class PromptsDataset(Dataset):
    """A dataset for loading prompts from a text file."""
    def __init__(self, prompts_file_path):
        self.prompts = self._load_prompts(prompts_file_path)

    def _load_prompts(self, file_path):
        logging.info(f"Loading prompts from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(prompts)} prompts.")
            return prompts
        except FileNotFoundError:
            logging.error(f"Prompts file not found at {file_path}")
            return []
        except Exception as e:
            logging.error(f"Error loading prompts from {file_path}: {e}")
            return []

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def get_prompt_loader(prompts_file_path, batch_size, shuffle=True):
    """Creates a DataLoader for the prompts dataset."""
    dataset = PromptsDataset(prompts_file_path)
    if len(dataset) == 0:
        raise ValueError(f"No prompts found in {prompts_file_path}. Please check the file.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    # A simple test to ensure the data loader works
    logging.basicConfig(level=logging.INFO)
    try:
        loader = get_prompt_loader(config.PROMPTS_FILE_PATH, batch_size=2)
        logging.info("DataLoader test successful. First batch:")
        for batch in loader:
            print(batch)
            break
    except Exception as e:
        logging.error(f"DataLoader test failed: {e}")