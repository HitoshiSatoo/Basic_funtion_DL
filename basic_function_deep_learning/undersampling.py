
import os
import random
import shutil

def balancear_dados_por_undersampling(dataset_path: str, output_path: str,class_dirs: list, max_images: int = None):
    """
    Função para balancear os dados de um dataset por undersampling, copiando imagens 
    para uma nova pasta com quantidade igual de imagens por classe.
    
    Parâmetros:
    - dataset_path (str): Caminho da pasta de imagens original.
    - output_path (str): Caminho da nova pasta para armazenar o dataset balanceado.
    - class_dirs (list): Lista de classes a serem balanceadas.
    - max_images (int, opcional): Quantidade máxima de imagens por classe. 
      Se None, será usado o número mínimo de imagens entre as classes do dataset original.
    """
    # Criar a nova pasta de saída se não existir
    os.makedirs(output_path, exist_ok=True)
    
    # Caminhos das classes
    class_paths = [os.path.join(dataset_path, class_dir) for class_dir in class_dirs]

    # Criar as subpastas na nova pasta de saída
    for class_dir in class_dirs:
        os.makedirs(os.path.join(output_path, class_dir), exist_ok=True)

    # Contar apenas arquivos com extensão .jpg e encontrar o número mínimo de imagens entre as classes
    class_image_counts = {class_dir: len([f for f in os.listdir(class_path) if f.endswith('.jpg')]) 
                          for class_dir, class_path in zip(class_dirs, class_paths)}

    # Definir o número de imagens a ser usado (mínimo entre as classes ou um valor especificado)
    if max_images is None:
        min_images = min(class_image_counts.values())
    else:
        min_images = min(max_images, min(class_image_counts.values()))

    print(f"Imagens por classe antes do balanceamento: {class_image_counts}")
    print(f"Cada classe terá {min_images} imagens após o balanceamento.")

    # Copiar as imagens selecionadas para a nova pasta de undersampling
    for class_dir, class_path in zip(class_dirs, class_paths):
        images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]  # Seleciona apenas arquivos .jpg
        
        # Selecionar aleatoriamente até o número mínimo de imagens por classe
        selected_images = random.sample(images, min_images)
        
        for image in selected_images:
            # Caminho completo da imagem de origem
            source = os.path.join(class_path, image)
            # Caminho para onde a imagem será copiada
            destination = os.path.join(output_path, class_dir, image)
            shutil.copy(source, destination)

    print(f"Dados balanceados e copiados para {output_path}. Cada classe agora tem {min_images} imagens.")


def balancear_varios_datasets(dataset_path: str, base_output_path: str, class_dirs: list, max_images_list: list):
    """
    Função para balancear os dados de um dataset em diferentes níveis de undersampling.
    
    Parâmetros:
    - dataset_path (str): Caminho da pasta de imagens original.
    - base_output_path (str): Caminho base da nova pasta para armazenar os datasets balanceados.
    - class_dirs (list): Lista com os nomes das subpastas que representam cada classe.
    - max_images_list (list): Lista com os valores de max_images para cada balanceamento.
    """
    for max_images in max_images_list:
        # Definir o caminho de saída com base no valor de max_images
        output_path = os.path.join(base_output_path, f"train_undersampling_{max_images}")
        
        # Chamar a função de balanceamento
        balancear_dados_por_undersampling(
            dataset_path=dataset_path,
            output_path=output_path,
            class_dirs=class_dirs,
            max_images=max_images
        )
        
        print(f"Concluído: Balanceamento para {max_images} imagens por classe.")
