
"""
Contém funcionalidades para criar DataLoaders do PyTorch para
dados de classificação de imagens.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                      test_dir: str,
                      transform: transforms.Compose,
                      batch_size: int,
                      num_workers: int=NUM_WORKERS):
  """
  Cria DataLoaders de treinamento e teste.

  Recebe um caminho para o diretório de treinamento e um caminho para
  o diretório de teste e os transforma em Datasets do PyTorch e depois em
  DataLoaders do PyTorch.

  Argumentos:
      train_dir: Caminho para o diretório de treinamento.
      test_dir: Caminho para o diretório de teste.
      transform: Transformações torchvision a serem aplicadas nos dados de treinamento e teste.
      batch_size: Número de amostras por lote em cada um dos DataLoaders.
      num_workers: Um inteiro para o número de trabalhadores por DataLoader.

  Retorna:
      Uma tupla de (train_dataloader, test_dataloader, class_names).
      Onde class_names é uma lista das classes de destino.
      Exemplo de uso:
        train_dataloader, test_dataloader, class_names = \

          = create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               transform=some_transform,
                               batch_size=32,
                               num_workers=4)
  """
  # Usar ImageFolder para criar dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Obter class names
  class_names = train_data.classes

  #
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
