
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Tuple, List, Dict

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Treina um modelo PyTorch por uma única época.

    Muda um modelo PyTorch alvo para o modo de treinamento e então
    executa todas as etapas necessárias de treinamento (passagem direta,
    cálculo de perda, passo do otimizador).

    Argumentos:
    model: Um modelo PyTorch a ser treinado.
    dataloader: Uma instância de DataLoader para o modelo ser treinado.
    loss_fn: Uma função de perda PyTorch a ser minimizada.
    optimizer: Um otimizador PyTorch para ajudar a minimizar a função de perda.
    device: Um dispositivo alvo para computar (por exemplo, "cuda" ou "cpu").

    Retorna:
    Uma tupla de métricas de perda e precisão de treinamento.
    Na forma (train_loss, train_accuracy). Por exemplo:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    """
    Testa um modelo PyTorch por uma única época.

    Muda um modelo PyTorch alvo para o modo "eval" e então realiza
    uma passagem direta em um conjunto de dados de teste.

    Argumentos:
    model: Um modelo PyTorch a ser testado.
    dataloader: Uma instância de DataLoader para o modelo ser testado.
    loss_fn: Uma função de perda PyTorch para calcular a perda nos dados de teste.
    device: Um dispositivo alvo para computar (por exemplo, "cuda" ou "cpu").

    Retorna:
    Uma tupla de métricas de perda e precisão de teste.
    Na forma (test_loss, test_accuracy). Por exemplo:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:

    """
    Treina e testa um modelo PyTorch.

    Passa um modelo PyTorch alvo pelas funções train_step() e test_step()
    por um número de épocas, treinando e testando o modelo
    no mesmo loop de época.

    Calcula, imprime e armazena métricas de avaliação ao longo do processo.

    Argumentos:
    model: Um modelo PyTorch a ser treinado e testado.
    train_dataloader: Uma instância de DataLoader para o modelo ser treinado.
    test_dataloader: Uma instância de DataLoader para o modelo ser testado.
    optimizer: Um otimizador PyTorch para ajudar a minimizar a função de perda.
    loss_fn: Uma função de perda PyTorch para calcular a perda em ambos os conjuntos de dados.
    epochs: Um inteiro indicando quantas épocas treinar.
    device: Um dispositivo alvo para computar (por exemplo, "cuda" ou "cpu").

    Retorna:
    Um dicionário de perda de treinamento e teste, bem como métricas de precisão de treinamento e teste. Cada métrica tem um valor em uma lista para
    cada época.
    Na forma: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    Por exemplo, se treinando por epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}

    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
