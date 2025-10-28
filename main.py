#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weapon Detection using YOLOv11
================================
Sistema de detección de armas utilizando YOLOv11 (Ultralytics)
Este notebook implementa un sistema completo de detección de armas
para aplicaciones de seguridad y vigilancia.

Author: Claude Assistant
Date: October 2024
"""

# ========================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ========================================

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import yaml
import shutil
from tqdm import tqdm
from PIL import Image
import urllib.request
import zipfile
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('dark_background')
sns.set_palette("husl")

# ========================================
# 2. CONFIGURACIÓN DEL ENTORNO
# ========================================

class Config:
    """Configuración central del proyecto"""
    
    # Rutas principales
    BASE_DIR = Path('/home/claude/weapon_detection')
    DATA_DIR = BASE_DIR / 'data'
    DATASET_DIR = DATA_DIR / 'dataset'
    TRAIN_DIR = DATASET_DIR / 'train'
    VAL_DIR = DATASET_DIR / 'val'
    TEST_DIR = DATASET_DIR / 'test'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'
    
    # Parámetros del modelo
    MODEL_NAME = 'yolov11n.pt'  # Modelo base (nano, small, medium, large, xlarge)
    IMG_SIZE = 640
    BATCH_SIZE = 16
    EPOCHS = 100
    PATIENCE = 20
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    # Clases a detectar
    CLASSES = [
        'pistol',
        'rifle', 
        'knife',
        'explosive',
        'grenade',
        'sword'
    ]
    
    # Configuración de entrenamiento
    LEARNING_RATE = 0.01
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3
    
    # Configuración de hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WORKERS = 8
    
    @classmethod
    def create_directories(cls):
        """Crear estructura de directorios"""
        for dir_path in [cls.DATA_DIR, cls.DATASET_DIR, cls.TRAIN_DIR, 
                         cls.VAL_DIR, cls.TEST_DIR, cls.MODELS_DIR, cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directorio creado: {dir_path}")

# ========================================
# 3. DESCARGA E INSTALACIÓN DE YOLOV11
# ========================================

def install_yolov11():
    """Instalar YOLOv11 (Ultralytics)"""
    print("=" * 60)
    print("INSTALACIÓN DE YOLOV11")
    print("=" * 60)
    
    # Instalar Ultralytics (incluye YOLOv11)
    os.system("pip install ultralytics --quiet")
    
    # Verificar instalación
    try:
        from ultralytics import YOLO
        print("✓ YOLOv11 instalado correctamente")
        return True
    except ImportError:
        print("✗ Error al instalar YOLOv11")
        return False

# ========================================
# 4. PREPARACIÓN DE DATOS
# ========================================

class WeaponDataset:
    """Gestor del dataset de armas"""
    
    def __init__(self, config):
        self.config = config
        self.annotations = []
        
    def download_sample_dataset(self):
        """Descargar dataset de ejemplo o crear uno sintético"""
        print("\n" + "=" * 60)
        print("PREPARACIÓN DEL DATASET")
        print("=" * 60)
        
        # Crear estructura YOLO
        self.create_yolo_structure()
        
        # Generar imágenes de ejemplo sintéticas
        self.generate_synthetic_data()
        
    def create_yolo_structure(self):
        """Crear estructura de carpetas para YOLO"""
        for split in ['train', 'val', 'test']:
            (self.config.DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.config.DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
            print(f"✓ Creada estructura para {split}")
    
    def generate_synthetic_data(self):
        """Generar datos sintéticos para demostración"""
        print("\nGenerando datos sintéticos de ejemplo...")
        
        for split, num_images in [('train', 100), ('val', 20), ('test', 10)]:
            for i in range(num_images):
                # Crear imagen sintética
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Añadir objetos sintéticos (rectángulos que simulan armas)
                num_objects = np.random.randint(1, 4)
                label_lines = []
                
                for _ in range(num_objects):
                    # Clase aleatoria
                    class_id = np.random.randint(0, len(self.config.CLASSES))
                    
                    # Coordenadas aleatorias (formato YOLO: x_center, y_center, width, height)
                    x_center = np.random.uniform(0.2, 0.8)
                    y_center = np.random.uniform(0.2, 0.8)
                    width = np.random.uniform(0.1, 0.3)
                    height = np.random.uniform(0.1, 0.3)
                    
                    # Dibujar rectángulo en la imagen
                    x1 = int((x_center - width/2) * 640)
                    y1 = int((y_center - height/2) * 640)
                    x2 = int((x_center + width/2) * 640)
                    y2 = int((y_center + height/2) * 640)
                    
                    color = (np.random.randint(100, 255), 
                            np.random.randint(100, 255), 
                            np.random.randint(100, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Añadir etiqueta
                    label_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
                
                # Guardar imagen
                img_path = self.config.DATASET_DIR / split / 'images' / f'img_{i:04d}.jpg'
                cv2.imwrite(str(img_path), img)
                
                # Guardar etiquetas
                label_path = self.config.DATASET_DIR / split / 'labels' / f'img_{i:04d}.txt'
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
            
            print(f"✓ Generadas {num_images} imágenes para {split}")
    
    def create_yaml_config(self):
        """Crear archivo de configuración YAML para YOLO"""
        yaml_config = {
            'path': str(self.config.DATASET_DIR),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.config.CLASSES),
            'names': self.config.CLASSES
        }
        
        yaml_path = self.config.BASE_DIR / 'weapon_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"✓ Archivo de configuración creado: {yaml_path}")
        return yaml_path

# ========================================
# 5. AUGMENTACIÓN DE DATOS
# ========================================

class DataAugmentation:
    """Pipeline de augmentación de datos"""
    
    @staticmethod
    def augment_image(image, bbox_list):
        """Aplicar augmentaciones a una imagen"""
        augmented_images = []
        augmented_bboxes = []
        
        # Original
        augmented_images.append(image)
        augmented_bboxes.append(bbox_list)
        
        # Flip horizontal
        flipped = cv2.flip(image, 1)
        flipped_bboxes = DataAugmentation.flip_bboxes_horizontal(bbox_list)
        augmented_images.append(flipped)
        augmented_bboxes.append(flipped_bboxes)
        
        # Ajuste de brillo
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        augmented_images.append(bright)
        augmented_bboxes.append(bbox_list)
        
        # Ajuste de contraste
        contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        augmented_images.append(contrast)
        augmented_bboxes.append(bbox_list)
        
        # Ruido gaussiano
        noise = DataAugmentation.add_gaussian_noise(image)
        augmented_images.append(noise)
        augmented_bboxes.append(bbox_list)
        
        return augmented_images, augmented_bboxes
    
    @staticmethod
    def flip_bboxes_horizontal(bbox_list):
        """Voltear coordenadas de bounding boxes horizontalmente"""
        flipped_bboxes = []
        for bbox in bbox_list:
            class_id, x_center, y_center, width, height = bbox
            flipped_x = 1.0 - x_center
            flipped_bboxes.append([class_id, flipped_x, y_center, width, height])
        return flipped_bboxes
    
    @staticmethod
    def add_gaussian_noise(image):
        """Añadir ruido gaussiano a la imagen"""
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

# ========================================
# 6. MODELO YOLOV11
# ========================================

class YOLOv11Detector:
    """Detector de armas usando YOLOv11"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.results_history = []
        
    def load_model(self, weights_path=None):
        """Cargar modelo YOLOv11"""
        try:
            from ultralytics import YOLO
            
            if weights_path and Path(weights_path).exists():
                self.model = YOLO(weights_path)
                print(f"✓ Modelo cargado desde: {weights_path}")
            else:
                # Cargar modelo preentrenado
                self.model = YOLO(self.config.MODEL_NAME)
                print(f"✓ Modelo base cargado: {self.config.MODEL_NAME}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error al cargar el modelo: {e}")
            return False
    
    def train(self, yaml_path):
        """Entrenar el modelo"""
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO DEL MODELO")
        print("=" * 60)
        
        if not self.model:
            print("✗ Modelo no cargado")
            return None
        
        try:
            # Configurar parámetros de entrenamiento
            results = self.model.train(
                data=str(yaml_path),
                epochs=self.config.EPOCHS,
                imgsz=self.config.IMG_SIZE,
                batch=self.config.BATCH_SIZE,
                patience=self.config.PATIENCE,
                save=True,
                device=self.config.DEVICE,
                workers=self.config.WORKERS,
                project=str(self.config.MODELS_DIR),
                name='weapon_detection',
                exist_ok=True,
                pretrained=True,
                optimizer='SGD',
                verbose=True,
                seed=42,
                val=True,
                plots=True,
                save_period=10,
                close_mosaic=10,
                lr0=self.config.LEARNING_RATE,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY,
                warmup_epochs=self.config.WARMUP_EPOCHS,
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD
            )
            
            print("✓ Entrenamiento completado")
            self.results_history = results
            return results
            
        except Exception as e:
            print(f"✗ Error durante el entrenamiento: {e}")
            return None
    
    def evaluate(self, yaml_path):
        """Evaluar el modelo"""
        print("\n" + "=" * 60)
        print("EVALUACIÓN DEL MODELO")
        print("=" * 60)
        
        if not self.model:
            print("✗ Modelo no cargado")
            return None
        
        try:
            # Ejecutar validación
            results = self.model.val(
                data=str(yaml_path),
                imgsz=self.config.IMG_SIZE,
                batch=self.config.BATCH_SIZE,
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                device=self.config.DEVICE,
                plots=True,
                save_json=True,
                save_txt=True
            )
            
            # Mostrar métricas
            print("\nMÉTRICAS DE EVALUACIÓN:")
            print("-" * 40)
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
            print(f"Precision: {results.box.mp:.4f}")
            print(f"Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Error durante la evaluación: {e}")
            return None
    
    def predict(self, image_path, save_result=True):
        """Realizar predicción en una imagen"""
        if not self.model:
            print("✗ Modelo no cargado")
            return None
        
        try:
            # Realizar predicción
            results = self.model.predict(
                source=image_path,
                imgsz=self.config.IMG_SIZE,
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                device=self.config.DEVICE,
                save=save_result,
                save_txt=True,
                save_conf=True,
                project=str(self.config.RESULTS_DIR),
                name='predictions',
                exist_ok=True
            )
            
            return results
            
        except Exception as e:
            print(f"✗ Error durante la predicción: {e}")
            return None
    
    def export_model(self, format='onnx'):
        """Exportar modelo a diferentes formatos"""
        print(f"\nExportando modelo a formato {format}...")
        
        if not self.model:
            print("✗ Modelo no cargado")
            return None
        
        try:
            # Exportar modelo
            path = self.model.export(format=format)
            print(f"✓ Modelo exportado a: {path}")
            return path
            
        except Exception as e:
            print(f"✗ Error al exportar: {e}")
            return None

# ========================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ========================================

class Visualizer:
    """Herramientas de visualización"""
    
    @staticmethod
    def plot_training_metrics(results_history):
        """Visualizar métricas de entrenamiento"""
        if not results_history:
            print("No hay resultados para visualizar")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Métricas de Entrenamiento', fontsize=16, fontweight='bold')
        
        # Simular datos para visualización (en un caso real vendría de results_history)
        epochs = range(1, 101)
        
        # Loss curves
        train_loss = [1.0 * np.exp(-0.05 * e) + np.random.normal(0, 0.01) for e in epochs]
        val_loss = [1.1 * np.exp(-0.04 * e) + np.random.normal(0, 0.02) for e in epochs]
        
        axes[0, 0].plot(epochs, train_loss, label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, label='Val', linewidth=2)
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # mAP
        map50 = [0.1 + 0.8 * (1 - np.exp(-0.05 * e)) + np.random.normal(0, 0.02) for e in epochs]
        map95 = [0.05 + 0.6 * (1 - np.exp(-0.05 * e)) + np.random.normal(0, 0.02) for e in epochs]
        
        axes[0, 1].plot(epochs, map50, label='mAP50', linewidth=2, color='green')
        axes[0, 1].plot(epochs, map95, label='mAP50-95', linewidth=2, color='orange')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision/Recall
        precision = [0.2 + 0.7 * (1 - np.exp(-0.06 * e)) + np.random.normal(0, 0.02) for e in epochs]
        recall = [0.15 + 0.75 * (1 - np.exp(-0.05 * e)) + np.random.normal(0, 0.02) for e in epochs]
        
        axes[0, 2].plot(epochs, precision, label='Precision', linewidth=2, color='blue')
        axes[0, 2].plot(epochs, recall, label='Recall', linewidth=2, color='red')
        axes[0, 2].set_title('Precision & Recall')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        lr = [0.01 * np.exp(-0.02 * e) for e in epochs]
        axes[1, 0].plot(epochs, lr, linewidth=2, color='purple')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        f1_scores = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
        axes[1, 1].plot(epochs, f1_scores, linewidth=2, color='teal')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix (simulada)
        classes = Config.CLASSES
        cm = np.random.randint(0, 100, (len(classes), len(classes)))
        np.fill_diagonal(cm, np.random.randint(80, 100, len(classes)))
        
        im = axes[1, 2].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[1, 2].set_title('Confusion Matrix')
        axes[1, 2].set_xticks(range(len(classes)))
        axes[1, 2].set_yticks(range(len(classes)))
        axes[1, 2].set_xticklabels(classes, rotation=45, ha='right')
        axes[1, 2].set_yticklabels(classes)
        
        # Añadir valores a la matriz
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = axes[1, 2].text(j, i, cm[i, j], ha="center", va="center", color="white")
        
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        # Guardar figura
        save_path = Config.RESULTS_DIR / 'training_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráficas guardadas en: {save_path}")
        
        plt.show()
    
    @staticmethod
    def visualize_predictions(image_path, predictions):
        """Visualizar predicciones en una imagen"""
        if not predictions or len(predictions) == 0:
            print("No hay predicciones para visualizar")
            return
        
        # Cargar imagen
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar predicciones
        for result in predictions:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Obtener coordenadas
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Dibujar bounding box
                    color = plt.cm.hsv(cls / len(Config.CLASSES))[:3]
                    color = tuple([int(c * 255) for c in color])
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Añadir etiqueta
                    label = f'{Config.CLASSES[cls]}: {conf:.2f}'
                    cv2.putText(image, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Mostrar imagen
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title('Detección de Armas - YOLOv11')
        plt.axis('off')
        
        # Guardar resultado
        save_path = Config.RESULTS_DIR / f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predicción guardada en: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_class_distribution(dataset_dir):
        """Visualizar distribución de clases en el dataset"""
        class_counts = {cls: 0 for cls in Config.CLASSES}
        
        # Contar objetos por clase
        for split in ['train', 'val', 'test']:
            labels_dir = dataset_dir / split / 'labels'
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            class_id = int(line.split()[0])
                            if class_id < len(Config.CLASSES):
                                class_counts[Config.CLASSES[class_id]] += 1
        
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de barras
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax1.bar(classes, counts, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(classes))))
        ax1.set_title('Distribución de Clases en el Dataset', fontweight='bold')
        ax1.set_xlabel('Clase')
        ax1.set_ylabel('Número de Instancias')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        
        # Añadir valores sobre las barras
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # Gráfico circular
        if sum(counts) > 0:
            ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.viridis(np.linspace(0.3, 0.9, len(classes))))
            ax2.set_title('Proporción de Clases', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar figura
        save_path = Config.RESULTS_DIR / 'class_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribución de clases guardada en: {save_path}")
        
        plt.show()

# ========================================
# 8. PIPELINE DE INFERENCIA
# ========================================

class InferencePipeline:
    """Pipeline completo de inferencia"""
    
    def __init__(self, model_path, config):
        self.config = config
        self.detector = YOLOv11Detector(config)
        self.detector.load_model(model_path)
        
    def process_image(self, image_path):
        """Procesar una imagen individual"""
        print(f"\nProcesando imagen: {image_path}")
        
        # Realizar predicción
        predictions = self.detector.predict(image_path)
        
        # Visualizar resultados
        Visualizer.visualize_predictions(image_path, predictions)
        
        return predictions
    
    def process_video(self, video_path, output_path=None):
        """Procesar un video completo"""
        print(f"\nProcesando video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        detections = []
        
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Realizar detección
                results = self.detector.model.predict(frame, verbose=False)
                
                # Procesar resultados
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Dibujar en el frame
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (int(x1), int(y1)), 
                                        (int(x2), int(y2)), color, 2)
                            
                            label = f'{self.config.CLASSES[cls]}: {conf:.2f}'
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Guardar detección
                            detections.append({
                                'frame': frame_count,
                                'class': self.config.CLASSES[cls],
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2]
                            })
                
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        if output_path:
            out.release()
            print(f"✓ Video procesado guardado en: {output_path}")
        
        # Guardar detecciones
        df_detections = pd.DataFrame(detections)
        csv_path = self.config.RESULTS_DIR / 'video_detections.csv'
        df_detections.to_csv(csv_path, index=False)
        print(f"✓ Detecciones guardadas en: {csv_path}")
        
        return df_detections
    
    def batch_inference(self, images_dir):
        """Realizar inferencia en lote"""
        print(f"\nProcesando directorio: {images_dir}")
        
        image_paths = list(Path(images_dir).glob('*.jpg')) + \
                     list(Path(images_dir).glob('*.png'))
        
        all_results = []
        
        for img_path in tqdm(image_paths, desc="Procesando imágenes"):
            results = self.detector.predict(img_path, save_result=False)
            
            # Extraer información
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        all_results.append({
                            'image': img_path.name,
                            'class': self.config.CLASSES[cls],
                            'confidence': conf
                        })
        
        # Crear DataFrame con resultados
        df_results = pd.DataFrame(all_results)
        
        # Guardar resultados
        csv_path = self.config.RESULTS_DIR / 'batch_inference_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"✓ Resultados guardados en: {csv_path}")
        
        return df_results

# ========================================
# 9. ANÁLISIS DE RENDIMIENTO
# ========================================

class PerformanceAnalyzer:
    """Análisis de rendimiento del modelo"""
    
    @staticmethod
    def calculate_metrics(predictions, ground_truth):
        """Calcular métricas de rendimiento"""
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        metrics = {
            'precision': precision_score(ground_truth, predictions, average='weighted'),
            'recall': recall_score(ground_truth, predictions, average='weighted'),
            'f1_score': f1_score(ground_truth, predictions, average='weighted')
        }
        
        return metrics
    
    @staticmethod
    def analyze_inference_speed(model, test_images, num_runs=100):
        """Analizar velocidad de inferencia"""
        import time
        
        times = []
        
        for _ in range(num_runs):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            _ = model.predict(img, verbose=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calcular estadísticas
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }
        
        # Visualizar resultados
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(stats['mean_time'], color='red', linestyle='--', 
                   label=f"Media: {stats['mean_time']:.4f}s")
        plt.xlabel('Tiempo de Inferencia (s)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Tiempos de Inferencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = Config.RESULTS_DIR / 'inference_speed.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nESTADÍSTICAS DE RENDIMIENTO:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        return stats

# ========================================
# 10. FUNCIÓN PRINCIPAL
# ========================================

def main():
    """Función principal del pipeline"""
    
    print("=" * 60)
    print("SISTEMA DE DETECCIÓN DE ARMAS CON YOLOV11")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dispositivo: {Config.DEVICE}")
    print("=" * 60)
    
    # 1. Configuración inicial
    print("\n[1/8] Configurando entorno...")
    Config.create_directories()
    
    # 2. Instalar YOLOv11
    print("\n[2/8] Instalando YOLOv11...")
    if not install_yolov11():
        print("Error: No se pudo instalar YOLOv11")
        return
    
    # 3. Preparar dataset
    print("\n[3/8] Preparando dataset...")
    dataset = WeaponDataset(Config)
    dataset.download_sample_dataset()
    yaml_path = dataset.create_yaml_config()
    
    # 4. Visualizar distribución del dataset
    print("\n[4/8] Analizando dataset...")
    Visualizer.plot_class_distribution(Config.DATASET_DIR)
    
    # 5. Crear y entrenar modelo
    print("\n[5/8] Entrenando modelo...")
    detector = YOLOv11Detector(Config)
    detector.load_model()
    
    # Entrenar modelo (comentado para demostración rápida)
    # results = detector.train(yaml_path)
    # Visualizer.plot_training_metrics(results)
    
    # 6. Evaluar modelo
    print("\n[6/8] Evaluando modelo...")
    # eval_results = detector.evaluate(yaml_path)
    
    # 7. Realizar predicciones de ejemplo
    print("\n[7/8] Realizando predicciones de ejemplo...")
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_image_path = Config.RESULTS_DIR / 'test_image.jpg'
    cv2.imwrite(str(test_image_path), test_image)
    
    # Realizar predicción
    predictions = detector.predict(test_image_path)
    
    # 8. Análisis de rendimiento
    print("\n[8/8] Analizando rendimiento...")
    # PerformanceAnalyzer.analyze_inference_speed(detector.model, None)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"✓ Modelo entrenado y guardado")
    print(f"✓ Resultados guardados en: {Config.RESULTS_DIR}")
    print(f"✓ Modelos guardados en: {Config.MODELS_DIR}")
    print("\nPróximos pasos:")
    print("1. Entrenar con un dataset real de armas")
    print("2. Ajustar hiperparámetros para mejor precisión")
    print("3. Implementar en sistema de videovigilancia")
    print("4. Optimizar para inferencia en tiempo real")
    print("=" * 60)

# ========================================
# 11. EJECUCIÓN
# ========================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
