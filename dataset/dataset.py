import os
import json
from typing import List, Dict, Any, Tuple
from PIL import Image
from torch.utils.data import Dataset
import numpy as np  

class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            # print(f'image_path:{image_path}')                        
            
            # # MK
            # image = image.resize((384,384)) # w,h
            
            # # MK
            # # ValueError: Unable to infer channel dimension format
            # # AttributeError: 'PngImageFile' object has no attribute 'ndim'
            if image.mode == 'RGBA': # Convert RGBA to RGB, some images were RGBA and raised above error             
                image = image.convert('RGB')                  
            
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")


class Region2DescDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = str(data['prefix'])
        suffix = str(data['suffix'])
        return prefix, suffix, image
    
class CaptionsDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = str(data['prefix'])
        suffix = str(data['suffix'])
        return prefix, suffix, image