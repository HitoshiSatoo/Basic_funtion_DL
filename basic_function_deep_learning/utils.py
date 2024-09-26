
"""
Contém varias funções de utilidades para modelos de treino e salvar para PyTorch
"""
from pathlib import Path

import torch

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Salva um modelo PyTorch em um diretório alvo.

    Argumentos:
    model: Um modelo PyTorch alvo a ser salvo.
    target_dir: Um diretório para salvar o modelo.
    model_name: Um nome de arquivo para o modelo salvo. Deve incluir
      ".pth" ou ".pt" como extensão de arquivo.

    Exemplo de uso:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
