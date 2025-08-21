from django.shortcuts import render, redirect
from django.db.models import Count
from django.http import JsonResponse
from .models import Emprendimiento, Municipio, Alcance, Tematica, Publicacion, Seguidores, Comentario, EmprendimientoTematica
import random
import json
import requests  
import os
from dotenv import load_dotenv
from django.shortcuts import render, redirect, get_object_or_404
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import logging
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from datetime import datetime
from sklearn.cluster import KMeans   
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import glob
from django.views.decorators.http import require_GET, require_POST
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_recall_fscore_support, accuracy_score
)
import math
import pymysql
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel



def home(request):
    return render(request, 'simulacion/home.html')

def lista_emprendimientos(request):
    # Obtener filtros desde los parámetros GET
    municipio_id = request.GET.get('municipio')
    alcance_id = request.GET.get('alcance')
    tematica_id = request.GET.get('tematica')
    sort_by = request.GET.get('sort_by', 'id_emprendimiento')  # Por defecto, ordenar por ID
    sort_order = request.GET.get('sort_order', 'asc')  # Por defecto, ascendente

    # Consulta base con optimización
    emprendimientos = Emprendimiento.objects.all().select_related('id_municipio_origen', 'id_alcance').prefetch_related('tematicas')

    # Aplicar filtros si existen
    if municipio_id:
        emprendimientos = emprendimientos.filter(id_municipio_origen__id_municipio=municipio_id)
    if alcance_id:
        emprendimientos = emprendimientos.filter(id_alcance__id_alcance=alcance_id)
    if tematica_id:
        emprendimientos = emprendimientos.filter(tematicas__id_tematica=tematica_id)

    # Aplicar ordenación
    if sort_by == 'id_emprendimiento':
        if sort_order == 'desc':
            emprendimientos = emprendimientos.order_by('-id_emprendimiento')
        else:
            emprendimientos = emprendimientos.order_by('id_emprendimiento')
    elif sort_by == 'municipio':
        if sort_order == 'desc':
            emprendimientos = emprendimientos.order_by('-id_municipio_origen__municipio')
        else:
            emprendimientos = emprendimientos.order_by('id_municipio_origen__municipio')

    # Obtener opciones para los filtros
    municipios = Municipio.objects.all()
    alcances = Alcance.objects.all()
    tematicas = Tematica.objects.all()

    context = {
        'emprendimientos': emprendimientos,
        'municipios': municipios,
        'alcances': alcances,
        'tematicas': tematicas,
        'sort_by': sort_by,
        'sort_order': sort_order,
    }
    return render(request, 'simulacion/emprendimientos.html', context)



def update_csv_files():
    """Función auxiliar para crear o sobrescribir los CSV con los datos de la base de datos."""
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATOS_DIR = os.path.join(BASE_DIR, 'DATOS')

        # Verificar que la carpeta DATOS existe
        if not os.path.exists(DATOS_DIR):
            os.makedirs(DATOS_DIR)  # Crear la carpeta si no existe

        # Crear o sobrescribir emprendimientos.csv
        emprendimientos_csv_path = os.path.join(DATOS_DIR, 'emprendimientos.csv')
        emprendimientos = Emprendimiento.objects.all()
        emprendimientos_data = [{
            'id_emprendimiento': emp.id_emprendimiento,
            'nombre_emprendimiento': emp.nombre_emprendimiento,
            'descripcion': emp.descripcion if emp.descripcion else '',
            'id_municipio_origen': emp.id_municipio_origen_id,
            'id_alcance': emp.id_alcance_id
        } for emp in emprendimientos]
        emprendimientos_df = pd.DataFrame(emprendimientos_data)
        if os.path.exists(emprendimientos_csv_path) and not os.access(emprendimientos_csv_path, os.W_OK):
            try:
                os.remove(emprendimientos_csv_path)  # Eliminar el archivo para recrearlo
            except Exception as e:
                raise Exception(f"No se puede escribir en {emprendimientos_csv_path}. Verifica si el archivo está abierto (por ejemplo, en Excel) o en modo de solo lectura: {str(e)}")
        emprendimientos_df.to_csv(emprendimientos_csv_path, encoding='utf-8-sig', index=False)

        # Crear o sobrescribir seguidores.csv
        seguidores_csv_path = os.path.join(DATOS_DIR, 'seguidores.csv')
        seguidores = Seguidores.objects.all()
        seguidores_data = [{
            'id_emprendimiento': seg.id_emprendimiento_id,
            'cantidad': seg.cantidad
        } for seg in seguidores]
        seguidores_df = pd.DataFrame(seguidores_data)
        if os.path.exists(seguidores_csv_path) and not os.access(seguidores_csv_path, os.W_OK):
            try:
                os.remove(seguidores_csv_path)
            except Exception as e:
                raise Exception(f"No se puede escribir en {seguidores_csv_path}. Verifica si el archivo está abierto o en modo de solo lectura: {str(e)}")
        seguidores_df.to_csv(seguidores_csv_path, encoding='utf-8-sig', index=False)

        # Crear o sobrescribir emprendimiento_tematica.csv
        emprendimiento_tematica_csv_path = os.path.join(DATOS_DIR, 'emprendimiento_tematica.csv')
        emprendimiento_tematica = EmprendimientoTematica.objects.values('id_emprendimiento_id', 'id_tematica_id')
        emprendimiento_tematica_data = [{
            'id_emprendimiento': et['id_emprendimiento_id'],
            'id_tematica': et['id_tematica_id']
        } for et in emprendimiento_tematica]
        emprendimiento_tematica_df = pd.DataFrame(emprendimiento_tematica_data)
        if os.path.exists(emprendimiento_tematica_csv_path) and not os.access(emprendimiento_tematica_csv_path, os.W_OK):
            try:
                os.remove(emprendimiento_tematica_csv_path)
            except Exception as e:
                raise Exception(f"No se puede escribir en {emprendimiento_tematica_csv_path}. Verifica si el archivo está abierto o en modo de solo lectura: {str(e)}")
        emprendimiento_tematica_df.to_csv(emprendimiento_tematica_csv_path, encoding='utf-8-sig', index=False)

        # Crear o sobrescribir publicaciones.csv
        publicaciones_csv_path = os.path.join(DATOS_DIR, 'publicaciones.csv')
        publicaciones = Publicacion.objects.all()
        publicaciones_data = [{
            'id_publicacion': pub.id_publicacion,
            'contenido': pub.contenido if pub.contenido else '',
            'n_likes': pub.n_likes,
            'id_emprendimiento': pub.id_emprendimiento_id
        } for pub in publicaciones]
        publicaciones_df = pd.DataFrame(publicaciones_data)
        if os.path.exists(publicaciones_csv_path) and not os.access(publicaciones_csv_path, os.W_OK):
            try:
                os.remove(publicaciones_csv_path)  # Eliminar el archivo para recrearlo
            except Exception as e:
                raise Exception(f"No se puede escribir en {publicaciones_csv_path}. Verifica si el archivo está abierto o en modo de solo lectura: {str(e)}")
        publicaciones_df.to_csv(publicaciones_csv_path, encoding='utf-8-sig', index=False)

        # Crear o sobrescribir comentarios.csv
        comentarios_csv_path = os.path.join(DATOS_DIR, 'comentarios.csv')
        comentarios = Comentario.objects.all()
        comentarios_data = [{
            'id_comentario': com.id_comentario,
            'comentario': com.comentario if com.comentario else '',
            'id_publicacion': com.id_publicacion_id
        } for com in comentarios]
        comentarios_df = pd.DataFrame(comentarios_data)
        if os.path.exists(comentarios_csv_path) and not os.access(comentarios_csv_path, os.W_OK):
            try:
                os.remove(comentarios_csv_path)  # Eliminar el archivo para recrearlo
            except Exception as e:
                raise Exception(f"No se puede escribir en {comentarios_csv_path}. Verifica si el archivo está abierto o en modo de solo lectura: {str(e)}")
        comentarios_df.to_csv(comentarios_csv_path, encoding='utf-8-sig', index=False)

    except Exception as e:
        raise Exception(f"Error al crear o sobrescribir los CSV: {str(e)}")

def agregar_emprendimiento(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)
    
    try:
        nombre_emprendimiento = request.POST.get('nombre_emprendimiento')
        descripcion = request.POST.get('descripcion', '')
        seguidores = request.POST.get('seguidores')
        id_municipio_origen = int(request.POST.get('id_municipio_origen'))
        id_alcance = int(request.POST.get('id_alcance'))
        tematicas = request.POST.getlist('tematicas')  # Lista de IDs de temáticas

        # Validar datos
        if not nombre_emprendimiento:
            return JsonResponse({'status': 'error', 'message': 'El nombre del emprendimiento es obligatorio'}, status=400)
        if not Municipio.objects.filter(id_municipio=id_municipio_origen).exists():
            return JsonResponse({'status': 'error', 'message': 'Municipio no válido'}, status=400)
        if not Alcance.objects.filter(id_alcance=id_alcance).exists():
            return JsonResponse({'status': 'error', 'message': 'Alcance no válido'}, status=400)
        if not tematicas:
            return JsonResponse({'status': 'error', 'message': 'Debe seleccionar al menos una temática'}, status=400)
        try:
            seguidores = int(seguidores)
            if seguidores < 0:
                return JsonResponse({'status': 'error', 'message': 'El número de seguidores no puede ser negativo'}, status=400)
        except (ValueError, TypeError):
            return JsonResponse({'status': 'error', 'message': 'El número de seguidores debe ser un número válido'}, status=400)

        # Crear emprendimiento en la base de datos
        emprendimiento = Emprendimiento.objects.create(
            nombre_emprendimiento=nombre_emprendimiento,
            descripcion=descripcion if descripcion else None,
            id_municipio_origen_id=id_municipio_origen,
            id_alcance_id=id_alcance
        )

        # Crear registro de seguidores
        Seguidores.objects.create(
            id_emprendimiento=emprendimiento,
            cantidad=seguidores
        )

        # Crear relaciones con temáticas
        for id_tematica in tematicas:
            if Tematica.objects.filter(id_tematica=id_tematica).exists():
                EmprendimientoTematica.objects.create(
                    id_emprendimiento=emprendimiento,
                    id_tematica=Tematica.objects.get(id_tematica=id_tematica)
                )

        # Crear o sobrescribir archivos CSV con los datos de la base de datos
        update_csv_files()

        # Limpiar SHARED_DATA para forzar recarga de datos
        SHARED_DATA.clear()

        return redirect('simulacion:lista_emprendimientos')
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Error al agregar emprendimiento: {str(e)}'}, status=500)

def modificar_emprendimiento(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)
    
    try:
        id_emprendimiento = int(request.POST.get('id_emprendimiento'))
        nombre_emprendimiento = request.POST.get('nombre_emprendimiento')
        descripcion = request.POST.get('descripcion', '')
        seguidores = request.POST.get('seguidores')
        id_municipio_origen = int(request.POST.get('id_municipio_origen'))
        id_alcance = int(request.POST.get('id_alcance'))
        tematicas = request.POST.getlist('tematicas')  # Lista de IDs de temáticas

        # Validar datos
        if not nombre_emprendimiento:
            return JsonResponse({'status': 'error', 'message': 'El nombre del emprendimiento es obligatorio'}, status=400)
        if not Municipio.objects.filter(id_municipio=id_municipio_origen).exists():
            return JsonResponse({'status': 'error', 'message': 'Municipio no válido'}, status=400)
        if not Alcance.objects.filter(id_alcance=id_alcance).exists():
            return JsonResponse({'status': 'error', 'message': 'Alcance no válido'}, status=400)
        if not tematicas:
            return JsonResponse({'status': 'error', 'message': 'Debe seleccionar al menos una temática'}, status=400)
        try:
            seguidores = int(seguidores)
            if seguidores < 0:
                return JsonResponse({'status': 'error', 'message': 'El número de seguidores no puede ser negativo'}, status=400)
        except (ValueError, TypeError):
            return JsonResponse({'status': 'error', 'message': 'El número de seguidores debe ser un número válido'}, status=400)

        # Actualizar emprendimiento en la base de datos
        try:
            emprendimiento = Emprendimiento.objects.get(id_emprendimiento=id_emprendimiento)
        except Emprendimiento.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Emprendimiento no encontrado'}, status=404)

        emprendimiento.nombre_emprendimiento = nombre_emprendimiento
        emprendimiento.descripcion = descripcion if descripcion else None
        emprendimiento.id_municipio_origen_id = id_municipio_origen
        emprendimiento.id_alcance_id = id_alcance
        emprendimiento.save()

        # Actualizar seguidores
        Seguidores.objects.filter(id_emprendimiento=emprendimiento).delete()
        Seguidores.objects.create(
            id_emprendimiento=emprendimiento,
            cantidad=seguidores
        )

        # Actualizar temáticas: eliminar las existentes y añadir las nuevas
        EmprendimientoTematica.objects.filter(id_emprendimiento=emprendimiento).delete()
        for id_tematica in tematicas:
            if Tematica.objects.filter(id_tematica=id_tematica).exists():
                EmprendimientoTematica.objects.create(
                    id_emprendimiento=emprendimiento,
                    id_tematica=Tematica.objects.get(id_tematica=id_tematica)
                )

        # Crear o sobrescribir archivos CSV con los datos de la base de datos
        update_csv_files()

        # Limpiar SHARED_DATA para forzar recarga de datos
        SHARED_DATA.clear()

        return redirect('simulacion:lista_emprendimientos')
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Error al modificar emprendimiento: {str(e)}'}, status=500)


def simulacion(request):
    # Obtener todos los emprendimientos con conteos de publicaciones y comentarios
    emprendimientos = Emprendimiento.objects.all().select_related('id_municipio_origen', 'id_alcance').prefetch_related('tematicas').annotate(
        num_publicaciones=Count('publicacion', distinct=True),
        num_comentarios=Count('publicacion__comentario', distinct=True)
    )
    context = {
        'emprendimientos': emprendimientos,
    }
    return render(request, 'simulacion/simulacion.html', context)



def guardar_publicaciones(request):
    if request.method == 'POST':
        id_emprendimiento = request.POST.get('id_emprendimiento')
        publicaciones_texto = request.POST.get('publicaciones', '')
        
        try:
            emprendimiento = Emprendimiento.objects.get(id_emprendimiento=id_emprendimiento)
            seguidores = Seguidores.objects.filter(id_emprendimiento=emprendimiento).first()
            num_seguidores = seguidores.cantidad if seguidores else 0

            # Procesar el texto para extraer las publicaciones
            publicaciones = [pub.strip() for pub in publicaciones_texto.split('),')]
            publicaciones = [pub.replace('(', '').replace(')', '').strip() for pub in publicaciones if pub.strip()]
            
            for contenido in publicaciones:
                # Calcular likes basados en seguidores
                caracteres = len(contenido)
                if num_seguidores == 0:
                    n_likes = random.randint(0, 5)
                else:
                    max_likes = min(num_seguidores // 10, 100)
                    n_likes = random.randint(0, max_likes)
                
                Publicacion.objects.create(
                    contenido=contenido,
                    n_likes=n_likes,
                    id_emprendimiento=emprendimiento
                )

            # Actualizar los archivos CSV después de guardar publicaciones
            update_csv_files()

            return redirect('simulacion:comentarios', id_emprendimiento=id_emprendimiento)
        except Emprendimiento.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Emprendimiento no encontrado'}, status=404)

def ver_publicaciones(request):
    if request.method == 'POST':
        id_emprendimiento = request.POST.get('id_emprendimiento')
        try:
            emprendimiento = get_object_or_404(Emprendimiento, id_emprendimiento=id_emprendimiento)
            publicaciones = Publicacion.objects.filter(id_emprendimiento=emprendimiento).annotate(
                num_comentarios=Count('comentario')
            ).values('id_publicacion', 'contenido', 'num_comentarios', 'n_likes')
            
            return JsonResponse({
                'status': 'success',
                'publicaciones': list(publicaciones),
                'emprendimiento': {
                    'id': emprendimiento.id_emprendimiento,
                    'nombre': emprendimiento.nombre_emprendimiento
                }
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

def publicaciones(request, id_emprendimiento):
    emprendimiento = get_object_or_404(Emprendimiento, id_emprendimiento=id_emprendimiento)
    return render(request, 'simulacion/publicaciones.html', {
        'emprendimiento': emprendimiento
    })

def comentarios(request, id_emprendimiento):
    emprendimiento = get_object_or_404(Emprendimiento, id_emprendimiento=id_emprendimiento)
    publicaciones = Publicacion.objects.filter(id_emprendimiento=emprendimiento).annotate(
        num_comentarios=Count('comentario')
    )
    return render(request, 'simulacion/comentarios.html', {
        'emprendimiento': emprendimiento,
        'publicaciones': publicaciones
    })


def agregar_comentarios(request, id_publicacion):
    publicacion = get_object_or_404(Publicacion, id_publicacion=id_publicacion)
    return render(request, 'simulacion/agregar_comentarios.html', {
        'publicacion': publicacion
    })

def guardar_comentarios(request):
    if request.method == 'POST':
        id_publicacion = request.POST.get('id_publicacion')
        comentarios_texto = request.POST.get('comentarios', '')
        
        try:
            publicacion = Publicacion.objects.get(id_publicacion=id_publicacion)
            
            # Procesar el texto para extraer los comentarios
            comentarios = [com.strip() for com in comentarios_texto.split('),')]
            comentarios = [com.replace('(', '').replace(')', '').strip() for com in comentarios if com.strip()]
            
            for contenido in comentarios:
                Comentario.objects.create(
                    comentario=contenido,
                    id_publicacion=publicacion
                )

            # Actualizar los archivos CSV después de guardar comentarios
            update_csv_files()

            return JsonResponse({'status': 'success', 'message': 'Comentarios guardados exitosamente'})
        except Publicacion.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Publicación no encontrada'}, status=404)
    
    return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

def ver_comentarios(request, id_publicacion):
    publicacion = get_object_or_404(Publicacion, id_publicacion=id_publicacion)
    comentarios = Comentario.objects.filter(id_publicacion=publicacion)
    return render(request, 'simulacion/ver_comentarios.html', {
        'publicacion': publicacion,
        'comentarios': comentarios
    })



"""

    EMBEDDINGS

"""


# Cargar modelo BETO
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Función para obtener embeddings de texto
def get_text_embedding(text):
    if not text or pd.isna(text):
        return np.zeros(768)  # Dimensión de BETO
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def check_missing_embeddings(request):
    """Verifica si hay emprendimientos nuevos (id > 80) sin embeddings generados."""
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'Embeddings', '512tk')
        EMBEDDINGS_SUBDIR = os.path.join(EMBEDDINGS_DIR, 'emprendimientos')
        
        # Crear la carpeta de emprendimientos si no existe
        if not os.path.exists(EMBEDDINGS_SUBDIR):
            os.makedirs(EMBEDDINGS_SUBDIR)
        
        # Filtrar solo emprendimientos con id > 80
        emprendimientos = Emprendimiento.objects.filter(id_emprendimiento__gt=80)
        emprendimientos_sin_embeddings = []
        all_embeddings_exist = True

        for emp in emprendimientos:
            embedding_file = os.path.join(EMBEDDINGS_SUBDIR, f'emprendimiento_{emp.id_emprendimiento}.txt')
            if not os.path.exists(embedding_file):
                emprendimientos_sin_embeddings.append({
                    'id_emprendimiento': emp.id_emprendimiento,
                    'nombre_emprendimiento': emp.nombre_emprendimiento
                })
                all_embeddings_exist = False

        return JsonResponse({
            'status': 'success',
            'all_embeddings_exist': all_embeddings_exist,
            'emprendimientos_sin_embeddings': emprendimientos_sin_embeddings
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def generar_embeddings(request):
    """Genera embeddings para emprendimientos nuevos (id > 80) y los guarda en archivos .npy."""
    if request.method == 'POST':
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'Embeddings', '512tk')
            EMBEDDINGS_SUBDIR = os.path.join(EMBEDDINGS_DIR, 'emprendimientos')
            
            # Crear la carpeta de emprendimientos si no existe
            if not os.path.exists(EMBEDDINGS_SUBDIR):
                os.makedirs(EMBEDDINGS_SUBDIR)
            
            # Filtrar solo emprendimientos con id > 80
            emprendimientos = Emprendimiento.objects.filter(id_emprendimiento__gt=80)
            emprendimientos_sin_embeddings = []
            missing_data = []

            # Verificar cuáles emprendimientos no tienen embeddings
            for emp in emprendimientos:
                embedding_file = os.path.join(EMBEDDINGS_SUBDIR, f'emprendimiento_{emp.id_emprendimiento}.txt')
                if not os.path.exists(embedding_file):
                    emprendimientos_sin_embeddings.append(emp)

            # Verificar si cada emprendimiento tiene publicaciones y comentarios
            for emp in emprendimientos_sin_embeddings:
                publicaciones = Publicacion.objects.filter(id_emprendimiento=emp).count()
                comentarios = Comentario.objects.filter(id_publicacion__id_emprendimiento=emp).count()
                if publicaciones == 0 or comentarios == 0:
                    missing_data.append({
                        'id_emprendimiento': emp.id_emprendimiento,
                        'nombre_emprendimiento': emp.nombre_emprendimiento,
                        'missing_publicaciones': publicaciones == 0,
                        'missing_comentarios': comentarios == 0
                    })

            if missing_data:
                message = "Los siguientes emprendimientos no tienen suficientes datos:\n"
                for emp in missing_data:
                    missing = []
                    if emp['missing_publicaciones']:
                        missing.append("publicaciones")
                    if emp['missing_comentarios']:
                        missing.append("comentarios")
                    message += f"- {emp['nombre_emprendimiento']} (falta: {', '.join(missing)})\n"
                return JsonResponse({'status': 'error', 'message': message}, status=400)

            # Generar embeddings para cada emprendimiento
            for emp in emprendimientos_sin_embeddings:
                # Generar embedding de la descripción
                descripcion = emp.descripcion if emp.descripcion else ""
                descripcion_embedding = get_text_embedding(descripcion)
                
                # Generar embeddings de las publicaciones
                publicaciones = Publicacion.objects.filter(id_emprendimiento=emp)
                contenido_embeddings = [get_text_embedding(pub.contenido) for pub in publicaciones]
                contenido_embeddings = np.array(contenido_embeddings) if contenido_embeddings else np.array([np.zeros(768)])
                
                # Generar embeddings de los comentarios
                comentarios = Comentario.objects.filter(id_publicacion__id_emprendimiento=emp)
                comentario_embeddings = [get_text_embedding(com.comentario) for com in comentarios]
                comentario_embeddings = np.array(comentario_embeddings) if comentario_embeddings else np.array([np.zeros(768)])
                
                # Guardar embeddings en archivos .npy
                np.save(os.path.join(EMBEDDINGS_SUBDIR, f'descripcion_{emp.id_emprendimiento}.npy'), descripcion_embedding)
                np.save(os.path.join(EMBEDDINGS_SUBDIR, f'contenido_{emp.id_emprendimiento}.npy'), contenido_embeddings)
                np.save(os.path.join(EMBEDDINGS_SUBDIR, f'comentario_{emp.id_emprendimiento}.npy'), comentario_embeddings)
                
                # Crear archivo .txt como marcador
                embedding_file = os.path.join(EMBEDDINGS_SUBDIR, f'emprendimiento_{emp.id_emprendimiento}.txt')
                with open(embedding_file, 'w') as f:
                    pass  # Crear archivo vacío

            return JsonResponse({'status': 'success', 'message': 'Embeddings generados exitosamente'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)












# Variable global para almacenar datos por modelo
SHARED_DATA = {}

def load_emb(path):
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        arr = np.vstack([np.array(x, dtype=np.float32) for x in arr])
    return arr

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_channels, 1)
        )
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
    def predict(self, x_src, x_dst):
        h = torch.cat([x_src, x_dst], dim=1)
        return torch.sigmoid(self.predictor(h)).view(-1)

def initialize_model_data(BASE_DIR, model_name, model_files):
    """Inicializa los datos del modelo y los almacena en SHARED_DATA."""
    if model_name not in model_files:
        return {'error': f'Modelo {model_name} no encontrado.'}

    if model_name in SHARED_DATA and SHARED_DATA[model_name].get('G') is not None:
        return SHARED_DATA[model_name]

    # ================== DATOS ==================
    emprendimientos = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/emprendimientos.csv'))
    publicaciones = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/publicaciones.csv'))
    comentarios = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/comentarios.csv'))
    seguidores = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/seguidores.csv'))
    emprendimiento_tematica = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/emprendimiento_tematica.csv'))
    municipios = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/municipios.csv'))
    tematicas = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/tematicas.csv'))
    
    print(f"Total emprendimientos en CSV: {len(emprendimientos)}")
    print(f"Total publicaciones en CSV: {len(publicaciones)}")
    print(f"Total comentarios en CSV: {len(comentarios)}")
    
    # ================== EMBEDDINGS ==================
    desc_embs = load_emb(os.path.join(BASE_DIR, 'embeddings/512tk/descripcion_embeddings.npy'))
    cont_embs = load_emb(os.path.join(BASE_DIR, 'embeddings/512tk/contenido_embeddings.npy'))
    comm_embs = load_emb(os.path.join(BASE_DIR, 'embeddings/512tk/comentario_embeddings.npy'))
    
    print(f"Dimensiones iniciales desc_embs: {desc_embs.shape}")
    print(f"Dimensiones iniciales cont_embs: {cont_embs.shape}")
    print(f"Dimensiones iniciales comm_embs: {comm_embs.shape}")
    
    new_desc, new_cont, new_comm = [], [], []
    new_desc_ids, new_cont_ids, new_comm_ids = [], [], []
    emb_dir = os.path.join(BASE_DIR, 'embeddings/512tk/emprendimientos')
    for f in os.listdir(emb_dir):
        if f.startswith("descripcion_") and f.endswith(".npy"):
            try:
                idx = int(f.replace("descripcion_", "").replace(".npy", ""))
                emb = load_emb(os.path.join(emb_dir, f))
                new_desc.append(emb)
                new_desc_ids.append(idx)
                print(f"Cargado embedding de descripción para id_emprendimiento {idx}")
            except ValueError:
                print(f"Nombre de archivo inválido: {f}")
        elif f.startswith("contenido_") and f.endswith(".npy"):
            try:
                idx = int(f.replace("contenido_", "").replace(".npy", ""))
                emb = load_emb(os.path.join(emb_dir, f))
                new_cont.append(emb)
                new_cont_ids.append(idx)
                print(f"Cargado embedding de contenido para id_publicacion {idx}")
            except ValueError:
                print(f"Nombre de archivo inválido: {f}")
        elif f.startswith("comentario_") and f.endswith(".npy"):
            try:
                idx = int(f.replace("comentario_", "").replace(".npy", ""))
                emb = load_emb(os.path.join(emb_dir, f))
                new_comm.append(emb)
                new_comm_ids.append(idx)
                print(f"Cargado embedding de comentario para id_comentario {idx}")
            except ValueError:
                print(f"Nombre de archivo inválido: {f}")
    
    if new_desc:
        new_desc = np.vstack(new_desc)
        desc_embs = np.vstack([desc_embs, new_desc])
        print(f"Dimensiones desc_embs después de agregar nuevos: {desc_embs.shape}")
    if new_cont:
        new_cont = np.vstack(new_cont)
        cont_embs = np.vstack([cont_embs, new_cont])
        print(f"Dimensiones cont_embs después de agregar nuevos: {cont_embs.shape}")
    if new_comm:
        new_comm = np.vstack(new_comm)
        comm_embs = np.vstack([comm_embs, new_comm])
        print(f"Dimensiones comm_embs después de agregar nuevos: {comm_embs.shape}")

    # Crear índices para todos los embeddings
    id_to_desc_idx = {emp_id: i for i, emp_id in enumerate(emprendimientos['id_emprendimiento'].values[:len(desc_embs)])}
    # Filtrar id_to_cont_idx para incluir solo IDs con embeddings válidos
    valid_cont_ids = set(new_cont_ids) | set(publicaciones['id_publicacion'].values[:len(cont_embs)])
    id_to_cont_idx = {pub_id: i for i, pub_id in enumerate(publicaciones['id_publicacion'].values)
                      if pub_id in valid_cont_ids}
    id_to_comm_idx = {com_id: i for i, com_id in enumerate(comentarios['id_comentario'].values[:len(comm_embs)])}

    print(f"Total IDs en id_to_desc_idx: {len(id_to_desc_idx)}")
    print(f"Total IDs en id_to_cont_idx: {len(id_to_cont_idx)}")
    print(f"Total IDs en id_to_comm_idx: {len(id_to_comm_idx)}")
    
    # Verificar correspondencia de índices
    for idx in new_cont_ids:
        if idx not in id_to_cont_idx:
            print(f"Advertencia: id_publicacion {idx} no encontrado en id_to_cont_idx")
    for idx in new_comm_ids:
        if idx not in id_to_comm_idx:
            print(f"Advertencia: id_comentario {idx} no encontrado en id_to_comm_idx")

    # ================== GRAFO ==================
    G = nx.Graph()
    valid_nodes = []
    for _, row in emprendimientos.iterrows():
        emp_id = row['id_emprendimiento']
        if emp_id in id_to_desc_idx:
            temas = emprendimiento_tematica[emprendimiento_tematica['id_emprendimiento'] == emp_id]['id_tematica'].tolist()
            G.add_node(emp_id, 
                       nombre_emprendimiento=row['nombre_emprendimiento'],
                       descripcion=row['descripcion'] if pd.notna(row['descripcion']) else '',
                       id_municipio_origen=row['id_municipio_origen'],
                       id_alcance=row['id_alcance'],
                       tematicas=temas)
            valid_nodes.append(emp_id)
        else:
            print(f"Advertencia: id_emprendimiento {emp_id} no tiene embedding de descripción, omitido del grafo")

    print(f"Total nodos válidos en el grafo: {len(valid_nodes)}")
    
    for _, pub in publicaciones.iterrows():
        id_emprendimiento = pub['id_emprendimiento']
        if id_emprendimiento not in valid_nodes:
            continue
        related_emprendimientos = emprendimiento_tematica[
            emprendimiento_tematica['id_tematica'].isin(
                emprendimiento_tematica[emprendimiento_tematica['id_emprendimiento'] == id_emprendimiento]['id_tematica']
            )
        ]['id_emprendimiento'].unique()
        related_emprendimientos = related_emprendimientos[related_emprendimientos != id_emprendimiento]
        for rel_emp in related_emprendimientos:
            if rel_emp in G.nodes:
                weight = pub['n_likes'] if pd.notna(pub['n_likes']) else 0
                G.add_edge(id_emprendimiento, rel_emp, weight=weight + 1)

    for emp1 in G.nodes:
        for emp2 in G.nodes:
            if emp1 < emp2:
                emp1_attrs = G.nodes[emp1]
                emp2_attrs = G.nodes[emp2]
                temas_comunes = len(set(emp1_attrs['tematicas']) & set(emp2_attrs['tematicas']))
                same_municipio = 1 if emp1_attrs['id_municipio_origen'] == emp2_attrs['id_municipio_origen'] else 0
                same_alcance = 1 if emp1_attrs['id_alcance'] == emp2_attrs['id_alcance'] else 0
                weight = temas_comunes + same_municipio + same_alcance
                if weight > 0:
                    G.add_edge(emp1, emp2, weight=weight)

    for _, row in seguidores.iterrows():
        id_emprendimiento = row['id_emprendimiento']
        if id_emprendimiento in G.nodes:
            G.nodes[id_emprendimiento]['seguidores'] = row['cantidad']

    # ================== FEATURES ==================
    scaler = MinMaxScaler()
    valid_emprendimientos = emprendimientos[emprendimientos['id_emprendimiento'].isin(valid_nodes)].copy()
    likes_por_emprendimiento = publicaciones[publicaciones['id_emprendimiento'].isin(valid_nodes)].groupby('id_emprendimiento')['n_likes'].sum().reset_index()
    likes_por_emprendimiento.columns = ['id_emprendimiento', 'total_likes']
    emprendimientos_features = valid_emprendimientos.merge(likes_por_emprendimiento, on='id_emprendimiento', how='left').fillna({'total_likes': 0})
    emprendimientos_features = emprendimientos_features.merge(
        seguidores[seguidores['id_emprendimiento'].isin(valid_nodes)][['id_emprendimiento', 'cantidad']], 
        on='id_emprendimiento', how='left').fillna({'cantidad': 0})
    numeric_features = scaler.fit_transform(emprendimientos_features[['total_likes', 'cantidad']])
    emprendimientos_features[['total_likes_norm', 'cantidad_norm']] = numeric_features

    print(f"Total emprendimientos_features: {len(emprendimientos_features)}")
    
    onehot_encoder_municipio = OneHotEncoder(sparse_output=False)
    municipio_encoded = onehot_encoder_municipio.fit_transform(emprendimientos_features[['id_municipio_origen']])
    municipio_encoded_df = pd.DataFrame(municipio_encoded, columns=onehot_encoder_municipio.get_feature_names_out(['id_municipio_origen']))

    onehot_encoder_alcance = OneHotEncoder(sparse_output=False)
    alcance_encoded = onehot_encoder_alcance.fit_transform(emprendimientos_features[['id_alcance']])
    alcance_encoded_df = pd.DataFrame(alcance_encoded, columns=onehot_encoder_alcance.get_feature_names_out(['id_alcance']))

    tematica_encoded = np.zeros((len(emprendimientos_features), len(tematicas)))
    valid_tematica = emprendimiento_tematica[emprendimiento_tematica['id_emprendimiento'].isin(valid_nodes)]
    for _, row in valid_tematica.iterrows():
        idx_emp = emprendimientos_features.index[emprendimientos_features['id_emprendimiento'] == row['id_emprendimiento']].tolist()
        if idx_emp:
            idx_emp = idx_emp[0]
            idx_tem = tematicas.index[tematicas['id_tematica'] == row['id_tematica']].tolist()[0]
            tematica_encoded[idx_emp, idx_tem] = 1
    tematica_encoded_df = pd.DataFrame(tematica_encoded, columns=[f'tematica_{i}' for i in tematicas['id_tematica']])

    features_df = pd.concat([emprendimientos_features[['id_emprendimiento', 'total_likes_norm', 'cantidad_norm']],
                             municipio_encoded_df, alcance_encoded_df, tematica_encoded_df], axis=1)

    print(f"Total filas en features_df: {len(features_df)}")
    
    for _, row in features_df.iterrows():
        if row['id_emprendimiento'] in G.nodes:
            G.nodes[row['id_emprendimiento']]['features'] = row.drop('id_emprendimiento').values

    # ================== PROCESAR EMBEDDINGS ==================
    W_DESC, W_PUB, W_COM = 0.5, 0.3, 0.2
    node_ids = sorted(G.nodes())
    emb_dim = desc_embs.shape[1]
    raw_text = np.zeros((len(node_ids), emb_dim), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        de = desc_embs[id_to_desc_idx[nid]]
        p_rows = publicaciones[publicaciones.id_emprendimiento == nid]
        if not p_rows.empty:
            pub_embs = []
            for pid in p_rows.id_publicacion.values:
                if pid in id_to_cont_idx and id_to_cont_idx[pid] < len(cont_embs):
                    pub_embs.append(cont_embs[id_to_cont_idx[pid]])
                else:
                    print(f"No se encontró embedding válido para id_publicacion={pid}")
            pe = np.nanmean(pub_embs, axis=0) if pub_embs else np.zeros(emb_dim, dtype=np.float32)
        else:
            pe = np.zeros(emb_dim, dtype=np.float32)

        pub_ids = p_rows.id_publicacion.values
        c_rows = comentarios[comentarios.id_publicacion.isin(pub_ids)]
        if not c_rows.empty:
            comm_embs_list = []
            for cid in c_rows.id_comentario.values:
                if cid in id_to_comm_idx and id_to_comm_idx[cid] < len(comm_embs):
                    comm_embs_list.append(comm_embs[id_to_comm_idx[cid]])
                else:
                    print(f"No se encontró embedding válido para id_comentario={cid}")
            ce = np.nanmean(comm_embs_list, axis=0) if comm_embs_list else np.zeros(emb_dim, dtype=np.float32)
        else:
            ce = np.zeros(emb_dim, dtype=np.float32)

        raw_text[i] = W_DESC * de + W_PUB * pe + W_COM * ce
        if nid == 81:
            print(f"Salud Vital (id=81): desc={de[:5]}, pub={pe[:5]}, comm={ce[:5]}, combined={raw_text[i][:5]}")

    print(f"Dimensiones de raw_text: {raw_text.shape}")
    
    # Calcular dimensiones de características numéricas
    dim_num = features_df.shape[1] - 1
    target_in_channels = 120
    n_components_svd = max(1, target_in_channels - dim_num)

    print(f"dim_num: {dim_num}, n_components_svd: {n_components_svd}")
    
    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
    text_feats = svd.fit_transform(raw_text)
    
    print(f"Dimensiones de text_feats: {text_feats.shape}, nodos esperados: {len(node_ids)}")
    
    if text_feats.shape[0] != len(node_ids):
        return {'error': f'Desajuste de dimensiones: text_feats tiene {text_feats.shape[0]} filas, esperado {len(node_ids)}'}
    
    for i, nid in enumerate(node_ids):
        if i >= text_feats.shape[0]:
            print(f"Error: Intento de acceso a índice {i} en text_feats con tamaño {text_feats.shape[0]}")
            break
        G.nodes[nid]['text_features'] = text_feats[i]

    # ================== COMBINAR FEATURES ==================
    num_list, has_num = [], []
    for nid in node_ids:
        f = G.nodes[nid].get('features')
        if f is not None:
            num_list.append(f)
            has_num.append(nid)
        else:
            print(f"Advertencia: id_emprendimiento {nid} no tiene features, usando ceros")
            num_list.append(np.zeros(dim_num, dtype=np.float32))
            has_num.append(nid)
    
    num_mat = np.vstack(num_list)
    scaler = MinMaxScaler().fit(num_mat)
    scaled = scaler.transform(num_mat)
    for j, nid in enumerate(has_num):
        G.nodes[nid]['scaled_num'] = scaled[j]

    A, B = 0.4, 0.6
    for nid in node_ids:
        num = G.nodes[nid].get('scaled_num', np.zeros(dim_num, dtype=np.float32))
        txt = G.nodes[nid]['text_features']
        G.nodes[nid]['combined_features'] = np.nan_to_num(np.hstack([A * num, B * txt]), nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor([G.nodes[n]['combined_features'] for n in node_ids], dtype=torch.float)
    if x.shape[1] != target_in_channels:
        return {'error': f'Dimensión de entrada incorrecta: se esperaba {target_in_channels}, pero se obtuvo {x.shape[1]}'}

    mapping = {node: idx for idx, node in enumerate(node_ids)}
    edges, weights = [], []
    for u, v, e in G.edges(data=True):
        if u in mapping and v in mapping:
            edges += [[mapping[u], mapping[v]], [mapping[v], mapping[u]]]
            weights += [e.get('weight', 1.0), e.get('weight', 1.0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_ids))

    # ================== MODELO ==================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(in_channels=target_in_channels, hidden_channels=256, out_channels=128).to(device)
    model_path = os.path.join(BASE_DIR, 'Modelo', model_name)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        return {'error': f'Error al cargar el modelo {model_name}: {str(e)}'}

    # ================== EMBEDDINGS GNN ==================
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()

    # ================== CLUSTERING ==================
    z = embeddings
    n_clusters = len(tematicas)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(z)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agg_labels = agglomerative.fit_predict(z)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(z)

    # ================== GUARDAR ==================
    SHARED_DATA[model_name] = {
        'G': G,
        'embeddings': embeddings,
        'node_ids': node_ids,
        'emprendimientos': emprendimientos,
        'emprendimiento_tematica': emprendimiento_tematica,
        'tematicas': tematicas,
        'municipios': municipios,
        'seguidores': seguidores,
        'publicaciones': publicaciones,
        'kmeans_labels': kmeans_labels,
        'agg_labels': agg_labels,
        'dbscan_labels': dbscan_labels
    }

    return SHARED_DATA[model_name]

def predicciones(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Obtener lista de modelos disponibles
    model_dir = os.path.join(BASE_DIR, 'Modelo')
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return render(request, 'simulacion/predicciones.html', {
            'error': 'No se encontraron modelos en la carpeta Modelo/'
        })
    
    # Seleccionar modelo por defecto (el más reciente basado en el nombre)
    default_model = max(model_files, key=lambda x: x.split('__')[-1].replace('.pth', ''))
    model_name = request.GET.get('model', default_model)
    
    # Inicializar datos del modelo
    data = initialize_model_data(BASE_DIR, model_name, model_files)
    if 'error' in data:
        return render(request, 'simulacion/predicciones.html', {
            'error': data['error']
        })

    G = data['G']
    emprendimientos = data['emprendimientos']
    emprendimiento_tematica = data['emprendimiento_tematica']
    tematicas = data['tematicas']
    municipios = data['municipios']
    embeddings = data['embeddings']
    node_ids = data['node_ids']
    seguidores = data['seguidores']
    publicaciones = data['publicaciones']
    kmeans_labels = data['kmeans_labels']
    agg_labels = data['agg_labels']
    dbscan_labels = data['dbscan_labels']

    # ================== GRAFO FRONTEND ==================
    partition = community_louvain.best_partition(G.to_undirected(), weight='weight', resolution=1.0)
    n_communities = len(set(partition.values()))

    graph_data = {'nodes': [], 'links': []}
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    for i, node in enumerate(node_ids):
        emp = emprendimientos[emprendimientos.id_emprendimiento == node]
        if not emp.empty:
            emp = emp.iloc[0]
            temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == node]['id_tematica'].tolist()
            tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
            seg = int(seguidores[seguidores.id_emprendimiento == node]['cantidad'].values[0]) if node in seguidores.id_emprendimiento.values else 0
            total_likes = int(publicaciones[publicaciones.id_emprendimiento == node]['n_likes'].sum()) if node in publicaciones.id_emprendimiento.values else 0
            graph_data['nodes'].append({
                'id': int(node),
                'nombre_emprendimiento': emp['nombre_emprendimiento'],
                'seguidores': seg,
                'total_likes': total_likes,
                'id_municipio_origen': int(emp['id_municipio_origen']),
                'municipio': municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else '',
                'tematicas': tema_names,
                'comunidad_louvain': int(partition[node]),
                'cluster_kmeans': int(kmeans_labels[i]),
                'cluster_agglomerative': int(agg_labels[i]),
                'cluster_dbscan': int(dbscan_labels[i]),
                'degree_centrality': float(degree_centrality[node]),
                'betweenness_centrality': float(betweenness_centrality[node])
            })
    for u, v, d in G.edges(data=True):
        if u in node_ids and v in node_ids:
            graph_data['links'].append({
                'source': int(u),
                'target': int(v),
                'weight': float(d.get('weight', 1.0))
            })

    emprendimientos_data = []
    for _, emp in emprendimientos.iterrows():
        temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == emp['id_emprendimiento']]['id_tematica'].tolist()
        tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
        seg = int(seguidores[seguidores.id_emprendimiento == emp['id_emprendimiento']]['cantidad'].values[0]) if emp['id_emprendimiento'] in seguidores.id_emprendimiento.values else 0
        emprendimientos_data.append({
            'id_emprendimiento': int(emp['id_emprendimiento']),
            'nombre_emprendimiento': emp['nombre_emprendimiento'],
            'descripcion': emp['descripcion'] if pd.notna(emp['descripcion']) else '',
            'municipio': municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else '',
            'seguidores': seg,
            'tematicas': tema_names
        })

    max_followers = int(seguidores['cantidad'].max()) if not seguidores.empty else 1000
    max_tematicas = int(emprendimiento_tematica.groupby('id_emprendimiento')['id_tematica'].count().max()) if not emprendimiento_tematica.empty else 1

    return render(request, 'simulacion/predicciones.html', {
        'graph_data': json.dumps(graph_data),
        'emprendimientos': json.dumps(emprendimientos_data),
        'tematicas': json.dumps([t['nombre'] for t in tematicas.to_dict('records')]),
        'municipios': municipios.to_dict('records'),
        'max_followers': max_followers,
        'max_tematicas': max_tematicas,
        'n_communities': n_communities,
        'model_files': model_files,
        'default_model': model_name
    })

def predicciones_data(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Obtener lista de modelos disponibles
    model_dir = os.path.join(BASE_DIR, 'Modelo')
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return JsonResponse({'status': 'error', 'message': 'No se encontraron modelos en la carpeta Modelo/'}, status=404)
    
    # Seleccionar modelo por defecto si no se proporciona
    model_name = request.GET.get('model')
    if not model_name:
        model_name = max(model_files, key=lambda x: x.split('__')[-1].replace('.pth', ''))

    model_path = os.path.join(BASE_DIR, 'Modelo', model_name)
    if not os.path.exists(model_path):
        return JsonResponse({'status': 'error', 'message': f'Modelo {model_name} no encontrado'}, status=404)

    # Inicializar datos del modelo
    data = initialize_model_data(BASE_DIR, model_name, model_files)
    if 'error' in data:
        return JsonResponse({'status': 'error', 'message': data['error']}, status=500)

    G = data['G']
    emprendimientos = data['emprendimientos']
    emprendimiento_tematica = data['emprendimiento_tematica']
    tematicas = data['tematicas']
    municipios = data['municipios']
    embeddings = data['embeddings']
    node_ids = data['node_ids']
    seguidores = data['seguidores']
    publicaciones = data['publicaciones']
    kmeans_labels = data['kmeans_labels']
    agg_labels = data['agg_labels']
    dbscan_labels = data['dbscan_labels']

    # Computar clustering nuevamente (por si cambió el grafo)
    z = embeddings
    n_clusters = len(tematicas)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(z)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agg_labels = agglomerative.fit_predict(z)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(z)

    partition = community_louvain.best_partition(G.to_undirected(), weight='weight', resolution=1.0)
    n_communities = len(set(partition.values()))

    graph_data = {
        'nodes': [],
        'links': []
    }
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    for i, node in enumerate(node_ids):
        emp = emprendimientos[emprendimientos.id_emprendimiento == node]
        if not emp.empty:
            emp = emp.iloc[0]
            temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == node]['id_tematica'].tolist()
            tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
            seg = int(seguidores[seguidores.id_emprendimiento == node]['cantidad'].values[0]) if node in seguidores.id_emprendimiento.values else 0
            total_likes = int(publicaciones[publicaciones.id_emprendimiento == node]['n_likes'].sum()) if node in publicaciones.id_emprendimiento.values else 0
            graph_data['nodes'].append({
                'id': int(node),
                'nombre_emprendimiento': emp['nombre_emprendimiento'],
                'seguidores': seg,
                'total_likes': total_likes,
                'id_municipio_origen': int(emp['id_municipio_origen']),
                'municipio': municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else '',
                'tematicas': tema_names,
                'comunidad_louvain': int(partition[node]),
                'cluster_kmeans': int(kmeans_labels[i]),
                'cluster_agglomerative': int(agg_labels[i]),
                'cluster_dbscan': int(dbscan_labels[i]),
                'degree_centrality': float(degree_centrality[node]),
                'betweenness_centrality': float(betweenness_centrality[node])
            })
    for u, v, d in G.edges(data=True):
        if u in node_ids and v in node_ids:
            graph_data['links'].append({
                'source': int(u),
                'target': int(v),
                'weight': float(d.get('weight', 1.0))
            })

    emprendimientos_data = []
    for _, emp in emprendimientos.iterrows():
        temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == emp['id_emprendimiento']]['id_tematica'].tolist()
        tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
        seg = int(seguidores[seguidores.id_emprendimiento == emp['id_emprendimiento']]['cantidad'].values[0]) if emp['id_emprendimiento'] in seguidores.id_emprendimiento.values else 0
        emprendimientos_data.append({
            'id_emprendimiento': int(emp['id_emprendimiento']),
            'nombre_emprendimiento': emp['nombre_emprendimiento'],
            'descripcion': emp['descripcion'] if pd.notna(emp['descripcion']) else '',
            'municipio': municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else '',
            'seguidores': seg,
            'tematicas': tema_names
        })

    return JsonResponse({
        'status': 'ok',
        'graph_data': graph_data,
        'emprendimientos': emprendimientos_data,
        'tematicas': [t['nombre'] for t in tematicas.to_dict('records')],
        'n_communities': n_communities
    })

def recommend_emprendimientos(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

    id_emprendimiento = int(request.POST.get('id_emprendimiento'))
    model_name = request.POST.get('model')
    
    if not model_name or model_name not in SHARED_DATA:
        return JsonResponse({'status': 'error', 'message': f'Modelo {model_name} no inicializado. Visita la página de predicciones primero.'}, status=400)

    # Usar datos preprocesados
    G = SHARED_DATA[model_name]['G']
    embeddings = SHARED_DATA[model_name]['embeddings']
    node_ids = SHARED_DATA[model_name]['node_ids']
    emprendimientos = SHARED_DATA[model_name]['emprendimientos']
    emprendimiento_tematica = SHARED_DATA[model_name]['emprendimiento_tematica']
    tematicas = SHARED_DATA[model_name]['tematicas']
    municipios = SHARED_DATA[model_name]['municipios']

    try:
        target_idx = np.where(np.array(node_ids) == id_emprendimiento)[0][0]
    except IndexError:
        return JsonResponse({'status': 'error', 'message': 'Emprendimiento no encontrado'}, status=404)

    from sklearn.preprocessing import normalize
    embeddings = normalize(embeddings)
    target_embedding = embeddings[target_idx].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = [i for i in sorted_indices if node_ids[i] != id_emprendimiento and similarities[i] > 0.1][:20]

    recommendations = []
    for idx in sorted_indices:
        emp_id = node_ids[idx]
        emp = emprendimientos[emprendimientos.id_emprendimiento == emp_id].iloc[0]
        temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == emp_id]['id_tematica'].tolist()
        tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
        recommendations.append({
            'id': int(emp_id),
            'nombre': emp['nombre_emprendimiento'],
            'descripcion': emp['descripcion'] if pd.notna(emp['descripcion']) else '',
            'municipio': municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else '',
            'alcance': int(emp['id_alcance']),
            'tematicas': tema_names,
            'similitud': f"{similarities[idx]:.3f}"
        })

    return JsonResponse({'status': 'success', 'recommendations': recommendations})

def generate_pdf_report(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)
    
    id_emprendimiento = int(request.POST.get('id_emprendimiento'))
    model_name = request.POST.get('model')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not model_name or model_name not in SHARED_DATA:
        return JsonResponse({
            'status': 'error',
            'message': f'Modelo {model_name} no inicializado. Visita la página de predicciones primero.'
        }, status=400)

    # Obtener datos del emprendimiento
    emprendimientos = SHARED_DATA[model_name]['emprendimientos']
    emprendimiento_tematica = SHARED_DATA[model_name]['emprendimiento_tematica']
    tematicas = SHARED_DATA[model_name]['tematicas']
    municipios = SHARED_DATA[model_name]['municipios']
    node_ids = SHARED_DATA[model_name]['node_ids']
    embeddings = SHARED_DATA[model_name]['embeddings']
    seguidores = SHARED_DATA[model_name]['seguidores']
    publicaciones = SHARED_DATA[model_name]['publicaciones']
    
    try:
        emp = emprendimientos[emprendimientos.id_emprendimiento == id_emprendimiento].iloc[0]
    except IndexError:
        return JsonResponse({
            'status': 'error',
            'message': f'Emprendimiento con ID {id_emprendimiento} no encontrado.'
        }, status=404)

    temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == id_emprendimiento]['id_tematica'].tolist()
    tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
    municipio = municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else ''
    seguidores_count = seguidores[seguidores.id_emprendimiento == id_emprendimiento]['cantidad'].values[0] if id_emprendimiento in seguidores.id_emprendimiento.values else 0
    total_likes = publicaciones[publicaciones.id_emprendimiento == id_emprendimiento]['n_likes'].sum() if id_emprendimiento in publicaciones.id_emprendimiento.values else 0
    
    # Obtener recomendaciones
    recommendations = []
    if id_emprendimiento in node_ids:
        try:
            target_idx = np.where(np.array(node_ids) == id_emprendimiento)[0][0]
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings)
            target_embedding = embeddings[target_idx].reshape(1, -1)
            similarities = cosine_similarity(target_embedding, embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_indices = [i for i in sorted_indices if node_ids[i] != id_emprendimiento and similarities[i] > 0.1][:20]
            
            for idx in sorted_indices:
                rec_id = node_ids[idx]
                rec_emp = emprendimientos[emprendimientos.id_emprendimiento == rec_id].iloc[0]
                rec_temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == rec_id]['id_tematica'].tolist()
                rec_tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in rec_temas]
                recommendations.append({
                    'nombre': rec_emp['nombre_emprendimiento'],
                    'similitud': f"{similarities[idx]:.3f}",
                    'tematicas': rec_tema_names
                })
        except IndexError:
            recommendations = []

    # Crear contexto para el template
    context = {
        'emprendimiento': {
            'nombre': emp['nombre_emprendimiento'],
            'descripcion': emp['descripcion'] if pd.notna(emp['descripcion']) else 'Sin descripción',
            'municipio': municipio,
            'seguidores': seguidores_count,
            'total_likes': total_likes,
            'tematicas': tema_names
        },
        'recommendations': recommendations,
        'fecha': datetime.now().strftime("%d/%m/%Y %H:%M")
    }
    
    # Renderizar template HTML
    template = get_template('simulacion/report_template.html')
    html = template.render(context)
    
    # Crear respuesta PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="reporte_{id_emprendimiento}.pdf"'
    
    # Generar PDF
    pisa_status = pisa.CreatePDF(html, dest=response)
    if pisa_status.err:
        return HttpResponse('Error al generar PDF', status=500)
    
    return response

"""

    FUNCIÓN DE LA PESTAÑA EVALUACIÓN


"""

# Si usas CPU en servidor, deja CPU. Si tienes CUDA disponible, el código lo soporta igual.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Nombre del modelo por defecto que mencionaste
DEFAULT_MODEL_NAME = "modelo_Entorno__20250813_224446.pth"

# Ruta base del app (la misma lógica que usas en predicciones)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Modelo")


# ---------- Helpers internos ----------

def _safe_torch_load(path):
    # Solo usar weights_only=False si confías en el origen del .pth
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    return ckpt

def _list_pth_models():
    """
    Lista los .pth en la carpeta Modelos para comparar versiones.
    """
    if not os.path.isdir(MODELS_DIR):
        return []
    files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    # Solo nombres (sin ruta)
    return sorted([os.path.basename(f) for f in files])

def _ensure_predictions_from_ckpt(ckpt):
    """
    Devuelve y_true (np.array), y_score (np.array en [0,1]) y, si está, el edge_label_index (pares de nodos) para grafo de errores.
    Intenta usar payload guardado; si no, reconstruye el modelo y predice sobre test_data dentro del checkpoint.
    """
    # 1) Si el checkpoint ya trae payload de evaluación, úsalo directo
    eval_payload = ckpt.get("eval_payload", {})
    y_true = eval_payload.get("y_true", None)
    y_score = eval_payload.get("y_score", None)
    edge_label_index = eval_payload.get("edge_label_index", None)  # shape (2, N) o lista de pares

    if y_true is not None and y_score is not None:
        y_true = np.array(y_true).astype(np.float32)
        y_score = np.array(y_score).astype(np.float32)
        if isinstance(edge_label_index, list):
            edge_label_index = np.array(edge_label_index, dtype=np.int64)
        return y_true, y_score, edge_label_index

    # 2) Si no hay payload, pero sí test_data, reconstruimos y predecimos
    test_data = ckpt.get("test_data", None)
    model_state = ckpt.get("model_state_dict", None)

    if test_data is None or model_state is None:
        raise RuntimeError(
            "El checkpoint no contiene 'eval_payload' ni 'test_data' + 'model_state_dict'. "
            "Guarda 'test_data' o 'eval_payload' al entrenar para habilitar este dashboard."
        )

    # Reconstruir el modelo GraphSAGE con dimensiones compatibles
    in_channels = test_data.x.shape[1]
    hidden_channels = 256
    out_channels = 128

    # Usa TU clase GraphSAGE definida arriba en tu views.py
    model = GraphSAGE(in_channels, hidden_channels, out_channels).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        h = model(test_data.x.to(DEVICE), test_data.edge_index.to(DEVICE))
        scores = model.predict(
            h[test_data.edge_label_index[0]], h[test_data.edge_label_index[1]]
        ).detach().cpu().numpy()

    y_true = test_data.edge_label.detach().cpu().numpy().astype(np.float32)
    y_score = scores.astype(np.float32)
    eli = test_data.edge_label_index.detach().cpu().numpy()  # (2, N)

    return y_true, y_score, eli

def _compute_all_metrics(y_true, y_score, threshold=0.5):
    """
    Calcula precisión, recall, f1, AUC, AP y accuracy con un umbral dado.
    """
    y_pred = (y_score >= float(threshold)).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "ap": float(ap),
        "accuracy": float(acc),
    }

def _curves(y_true, y_score):
    """
    Devuelve ROC (fpr, tpr, thresholds) y PR (precision, recall, thresholds) como listas JSON-serializables.
    """
    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    precision, recall, pr_th = precision_recall_curve(y_true, y_score)
    return {
        "roc": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in roc_th],
        },
        "pr": {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "thresholds": [float(x) for x in pr_th] if pr_th is not None and len(pr_th) > 0 else [],
        },        
    }

def _histogram(y_score, bins=20):
    """
    Histograma simple en [0,1] para Chart.js tipo 'bar'.
    """
    if len(y_score) == 0:
        print("Advertencia: y_score está vacío, devolviendo histograma vacío")
        return {"bin_centers": [], "counts": []}
    counts, edges = np.histogram(y_score, bins=min(bins, 20), range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    print("Histogram bins:", len(centers), "Counts:", counts) # Debug
    return {
        "bin_centers": [float(x) for x in centers],
        "counts": [int(c) for c in counts]
    }

def _top_errors(y_true, y_score, edge_label_index=None, top_k=30):
    """
    Extrae las top FP y top FN más "confiadas" (FP con score mayor; FN con score menor).
    Devuelve lista con pares de nodos y score/label para mostrar y graficar mini-grafo.
    """
    y_true = y_true.astype(int)
    idx_all = np.arange(len(y_true))

    # FP: y_true=0, y_score alto
    fp_mask = (y_true == 0)
    fp_idx = idx_all[fp_mask]
    fp_scores = y_score[fp_mask]
    order_fp = np.argsort(fp_scores)[::-1]
    top_fp_idx = fp_idx[order_fp][:top_k]

    # FN: y_true=1, y_score bajo
    fn_mask = (y_true == 1)
    fn_idx = idx_all[fn_mask]
    fn_scores = y_score[fn_mask]
    order_fn = np.argsort(fn_scores)  # de menor a mayor
    top_fn_idx = fn_idx[order_fn][:top_k]

    def pack(indices, label):
        out = []
        for i in indices:
            item = {"idx": int(i), "y_true": int(y_true[i]), "score": float(y_score[i]), "type": label}
            if edge_label_index is not None and isinstance(edge_label_index, np.ndarray) and edge_label_index.shape[0] == 2:
                u = int(edge_label_index[0, i])
                v = int(edge_label_index[1, i])
                item["u"] = u
                item["v"] = v
            out.append(item)
        return out

    fp_list = pack(top_fp_idx, "FP")
    fn_list = pack(top_fn_idx, "FN")

    # Grafo pequeño: nodos = únicos u,v de errores; enlaces = esos pares
    nodes_set = set()
    links = []
    for item in fp_list + fn_list:
        u = item.get("u")
        v = item.get("v")
        if u is not None and v is not None:
            nodes_set.add(u)
            nodes_set.add(v)
            links.append({"source": u, "target": v, "etype": item["type"], "score": item["score"]})

    nodes = [{"id": int(n)} for n in nodes_set]
    return {
        "fp": fp_list,
        "fn": fn_list,
        "error_graph": {"nodes": nodes, "links": links}
    }

@require_GET
def evaluacion(request):
    """
    Renderiza la página; los datos se piden por AJAX a /evaluacion/data/.
    """
    models = _list_pth_models()
    # Asegura que el modelo por defecto esté en la lista (por si acaso)
    if DEFAULT_MODEL_NAME not in models:
        models.insert(0, DEFAULT_MODEL_NAME)

    return render(request, "simulacion/evaluacion.html", {
        "default_model": DEFAULT_MODEL_NAME,
        "model_files": models,
    })

@require_GET
def evaluacion_data(request):
    """
    Devuelve en JSON todas las métricas/calculables para inicializar el dashboard.
    Acepta ?model=<nombre.pth> opcional.
    """
    model_name = request.GET.get("model", DEFAULT_MODEL_NAME)
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.isfile(model_path):
        return JsonResponse(clean_for_json({
            "status": "error",
            "message": f"No existe el modelo: {model_name}"
        }), status=404)

    try:
        ckpt = _safe_torch_load(model_path)
        history = ckpt.get("history", {})
        test_metrics = ckpt.get("test_metrics", {})

        # Garantizar y_true / y_score / edge_label_index
        y_true, y_score, edge_label_index = _ensure_predictions_from_ckpt(ckpt)

        # Curvas y métricas con threshold 0.5 inicial
        curves = _curves(y_true, y_score)
        metrics = _compute_all_metrics(y_true, y_score, threshold=0.5)
        hist = _histogram(y_score, bins=20)
        errors = _top_errors(y_true, y_score, edge_label_index=edge_label_index, top_k=30)

        # History saneado (por si no existe alguna métrica)
        hist_out = {
            "epochs": list(range(1, len(history.get("train_loss", [])) + 1)),
            "train_loss": history.get("train_loss", []),
            "val_auc": history.get("val_metrics", {}).get("auc", []),
            "val_f1": history.get("val_metrics", {}).get("f1", []),
            "val_precision": history.get("val_metrics", {}).get("precision", []),
            "val_recall": history.get("val_metrics", {}).get("recall", []),
        }

        # Formato extra para el frontend (evitar undefined.epochs)
        f1_history = {
            "epochs": hist_out["epochs"],
            "values": hist_out["val_f1"]
        }
        loss_history = {
            "epochs": hist_out["epochs"],
            "values": hist_out["train_loss"]
        }

        return JsonResponse(clean_for_json({
            "status": "ok",
            "model_name": model_name,
            "history": hist_out,
            "f1_history": f1_history,        
            "loss_history": loss_history,    
            "test_metrics_saved": test_metrics,
            "metrics": metrics,
            "roc": curves["roc"],
            "pr": curves["pr"],
            "hist": hist,
            "errors": {
                "fp": errors["fp"],
                "fn": errors["fn"]
            },
            "error_graph": errors["error_graph"]
        }))

    except Exception as e:
        return JsonResponse(clean_for_json({
            "status": "error",
            "message": repr(e)
        }), status=500)

@require_POST
def recalculate_metrics(request):
    """
    Recalcula con un umbral nuevo. Recibe JSON:
    { "threshold": 0.42, "model": "..." }
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
        threshold = float(body.get("threshold", 0.5))
        model_name = body.get("model", DEFAULT_MODEL_NAME)
    except Exception:
        threshold = 0.5
        model_name = DEFAULT_MODEL_NAME

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isfile(model_path):
        return JsonResponse(clean_for_json({
            "status": "error",
            "message": f"No existe el modelo: {model_name}"
        }), status=404)

    try:
        ckpt = _safe_torch_load(model_path)
        y_true, y_score, _ = _ensure_predictions_from_ckpt(ckpt)
        metrics = _compute_all_metrics(y_true, y_score, threshold=threshold)
        return JsonResponse(clean_for_json({
            "status": "ok",
            "metrics": metrics,
            "threshold": threshold
        }))
    except Exception as e:
        return JsonResponse(clean_for_json({
            "status": "error",
            "message": repr(e)
        }), status=500)

@require_GET
def model_summary(request):
    """
    Devuelve resumen de un modelo dado para comparación (métricas base a threshold=0.5).
    Acepta ?model=<nombre.pth>
    """
    model_name = request.GET.get("model", DEFAULT_MODEL_NAME)
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isfile(model_path):
        return JsonResponse(clean_for_json({
            "status": "error",
            "message": f"No existe el modelo: {model_name}"
        }), status=404)

    try:
        ckpt = _safe_torch_load(model_path)
        test_metrics_saved = ckpt.get("test_metrics", {})
        y_true, y_score, _ = _ensure_predictions_from_ckpt(ckpt)
        metrics_05 = _compute_all_metrics(y_true, y_score, threshold=0.5)
        return JsonResponse(clean_for_json({
            "status": "ok",
            "model_name": model_name,
            "metrics_saved": test_metrics_saved,
            "metrics_05": metrics_05
        }))
    except Exception as e:
        return JsonResponse(clean_for_json({
            "status": "error",
            "message": repr(e)
        }), status=500)
    

def clean_for_json(obj):
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None  # o un valor máximo permitido
        return obj
    else:
        return obj