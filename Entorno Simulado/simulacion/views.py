from django.shortcuts import render, redirect
from django.db.models import Count
from django.http import JsonResponse
from .models import Emprendimiento, Municipio, Alcance, Tematica, Publicacion, Seguidores, Comentario
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
from datetime import datetime  # Agrega esta línea al inicio con las otras importaciones





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

            return redirect('simulacion:comentarios', id_emprendimiento=id_emprendimiento)
        except Emprendimiento.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Emprendimiento no encontrado'}, status=404)
    
    return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

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

# Variable global para almacenar datos preprocesados
SHARED_DATA = {
    'G': None,
    'embeddings': None,
    'node_ids': None,
    'emprendimientos': None,
    'emprendimiento_tematica': None,
    'tematicas': None
}

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

def predicciones(request):
    global SHARED_DATA
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Cargar datos
    emprendimientos = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/emprendimientos.csv'))
    publicaciones = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/publicaciones.csv'))
    comentarios = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/comentarios.csv'))
    seguidores = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/seguidores.csv'))
    emprendimiento_tematica = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/emprendimiento_tematica.csv'))
    municipios = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/municipios.csv'))
    tematicas = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/tematicas.csv'))
    desc_embs = load_emb(os.path.join(BASE_DIR, 'Embeddings/512tk/descripcion_embeddings.npy'))
    cont_embs = load_emb(os.path.join(BASE_DIR, 'Embeddings/512tk/contenido_embeddings.npy'))
    comm_embs = load_emb(os.path.join(BASE_DIR, 'Embeddings/512tk/comentario_embeddings.npy'))

    # Crear grafo
    G = nx.Graph()
    for _, row in emprendimientos.iterrows():
        temas = emprendimiento_tematica[emprendimiento_tematica['id_emprendimiento'] == row['id_emprendimiento']]['id_tematica'].tolist()
        G.add_node(row['id_emprendimiento'], 
                   nombre_emprendimiento=row['nombre_emprendimiento'],
                   descripcion=row['descripcion'] if pd.notna(row['descripcion']) else '',
                   id_municipio_origen=row['id_municipio_origen'],
                   id_alcance=row['id_alcance'],
                   tematicas=temas)

    for _, pub in publicaciones.iterrows():
        id_emprendimiento = pub['id_emprendimiento']
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
            for emp2 in G.neighbors(id_emprendimiento):
                G[id_emprendimiento][emp2]['weight'] += row['cantidad'] / 1000

    # Procesar características
    scaler = MinMaxScaler()
    likes_por_emprendimiento = publicaciones.groupby('id_emprendimiento')['n_likes'].sum().reset_index()
    likes_por_emprendimiento.columns = ['id_emprendimiento', 'total_likes']
    emprendimientos_features = emprendimientos.merge(likes_por_emprendimiento, on='id_emprendimiento', how='left').fillna({'total_likes': 0})
    emprendimientos_features = emprendimientos_features.merge(seguidores[['id_emprendimiento', 'cantidad']], on='id_emprendimiento', how='left').fillna({'cantidad': 0})
    numeric_features = scaler.fit_transform(emprendimientos_features[['total_likes', 'cantidad']])
    emprendimientos_features[['total_likes_norm', 'cantidad_norm']] = numeric_features

    onehot_encoder_municipio = OneHotEncoder(sparse_output=False)
    municipio_encoded = onehot_encoder_municipio.fit_transform(emprendimientos[['id_municipio_origen']])
    municipio_encoded_df = pd.DataFrame(municipio_encoded, columns=onehot_encoder_municipio.get_feature_names_out(['id_municipio_origen']))

    onehot_encoder_alcance = OneHotEncoder(sparse_output=False)
    alcance_encoded = onehot_encoder_alcance.fit_transform(emprendimientos[['id_alcance']])
    alcance_encoded_df = pd.DataFrame(alcance_encoded, columns=onehot_encoder_alcance.get_feature_names_out(['id_alcance']))

    tematica_encoded = np.zeros((len(emprendimientos), len(tematicas)))
    for _, row in emprendimiento_tematica.iterrows():
        idx_emp = emprendimientos.index[emprendimientos['id_emprendimiento'] == row['id_emprendimiento']].tolist()[0]
        idx_tem = tematicas.index[tematicas['id_tematica'] == row['id_tematica']].tolist()[0]
        tematica_encoded[idx_emp, idx_tem] = 1
    tematica_encoded_df = pd.DataFrame(tematica_encoded, columns=[f'tematica_{i}' for i in tematicas['id_tematica']])

    features_df = pd.concat([emprendimientos_features[['id_emprendimiento', 'total_likes_norm', 'cantidad_norm']],
                             municipio_encoded_df, alcance_encoded_df, tematica_encoded_df], axis=1)

    for _, row in features_df.iterrows():
        if row['id_emprendimiento'] in G.nodes:
            G.nodes[row['id_emprendimiento']]['features'] = row.drop('id_emprendimiento').values

    # Procesar embeddings
    W_DESC, W_PUB, W_COM = 0.5, 0.3, 0.2
    node_ids = sorted(G.nodes())
    emb_dim = desc_embs.shape[1]
    raw_text = np.zeros((len(node_ids), emb_dim), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        row = emprendimientos[emprendimientos.id_emprendimiento == nid]
        de = desc_embs[row.index[0]] if not row.empty else np.zeros(emb_dim, dtype=np.float32)
        p_rows = publicaciones[publicaciones.id_emprendimiento == nid]
        pe = np.nanmean(cont_embs[p_rows.index.values], axis=0) if not p_rows.empty else np.zeros(emb_dim, dtype=np.float32)
        pub_ids = p_rows.id_publicacion.values
        c_rows = comentarios[comentarios.id_publicacion.isin(pub_ids)]
        ce = np.nanmean(comm_embs[c_rows.index.values], axis=0) if not c_rows.empty else np.zeros(emb_dim, dtype=np.float32)
        raw_text[i] = W_DESC * de + W_PUB * pe + W_COM * ce

    svd = TruncatedSVD(n_components=128, random_state=42)
    text_feats = svd.fit_transform(raw_text)
    for i, nid in enumerate(node_ids):
        G.nodes[nid]['text_features'] = text_feats[i]

    # Combinar características
    num_list, has_num = [], []
    for nid in node_ids:
        f = G.nodes[nid].get('features')
        if f is not None:
            num_list.append(f)
            has_num.append(nid)
    if num_list:
        num_mat = np.vstack(num_list)
        scaler = MinMaxScaler().fit(num_mat)
        scaled = scaler.transform(num_mat)
        for j, nid in enumerate(has_num):
            G.nodes[nid]['scaled_num'] = scaled[j]

    A, B = 0.4, 0.6
    for nid in node_ids:
        num = G.nodes[nid].get('scaled_num', np.zeros_like(text_feats[0], dtype=np.float32))
        txt = G.nodes[nid]['text_features']
        G.nodes[nid]['combined_features'] = np.nan_to_num(np.hstack([A * num, B * txt]), nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor([G.nodes[n]['combined_features'] for n in node_ids], dtype=torch.float)
    mapping = {node: idx for idx, node in enumerate(node_ids)}
    edges, weights = [], []
    for u, v, e in G.edges(data=True):
        if u in mapping and v in mapping:
            edges += [[mapping[u], mapping[v]], [mapping[v], mapping[u]]]
            weights += [e.get('weight', 1.0), e.get('weight', 1.0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_ids))

    # Cargar modelo GraphSAGE
    device = torch.device('cpu')
    in_channels = x.shape[1]
    model = GraphSAGE(in_channels=in_channels, hidden_channels=256, out_channels=128).to(device)
    model_path = os.path.join(BASE_DIR, 'Modelo/model_5_20250627_184503.pth')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generar embeddings
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()

    # Almacenar datos para recommend_emprendimientos
    SHARED_DATA['G'] = G
    SHARED_DATA['embeddings'] = embeddings
    SHARED_DATA['node_ids'] = node_ids
    SHARED_DATA['emprendimientos'] = emprendimientos
    SHARED_DATA['emprendimiento_tematica'] = emprendimiento_tematica
    SHARED_DATA['tematicas'] = tematicas

    # Crear datos para el frontend
    partition = community_louvain.best_partition(G.to_undirected(), weight='weight', resolution=1.0)
    n_communities = len(set(partition.values()))

    graph_data = {
        'nodes': [],
        'links': []
    }
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    for node in G.nodes():
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
                'degree_centrality': float(degree_centrality[node]),
                'betweenness_centrality': float(betweenness_centrality[node])
            })
    for u, v, d in G.edges(data=True):
        if u in mapping and v in mapping:
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
        'n_communities': n_communities
    })

def recommend_emprendimientos(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)

    id_emprendimiento = int(request.POST.get('id_emprendimiento'))
    
    # Verificar que los datos estén inicializados
    if not all(v is not None for v in SHARED_DATA.values()):
        return JsonResponse({'status': 'error', 'message': 'Datos no inicializados. Visita la página de predicciones primero.'}, status=500)

    # Usar datos preprocesados
    G = SHARED_DATA['G']
    embeddings = SHARED_DATA['embeddings']
    node_ids = SHARED_DATA['node_ids']
    emprendimientos = SHARED_DATA['emprendimientos']
    emprendimiento_tematica = SHARED_DATA['emprendimiento_tematica']
    tematicas = SHARED_DATA['tematicas']

    try:
        target_idx = np.where(np.array(node_ids) == id_emprendimiento)[0][0]
    except IndexError:
        return JsonResponse({'status': 'error', 'message': 'Emprendimiento no encontrado'}, status=404)

    target_embedding = embeddings[target_idx].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = [i for i in sorted_indices if node_ids[i] != id_emprendimiento][:5]

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
            'tematicas': tema_names,
            'similitud': float(similarities[idx])
        })

    return JsonResponse({'status': 'success', 'recommendations': recommendations})


def generate_pdf_report(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Método no permitido'}, status=405)
    
    id_emprendimiento = int(request.POST.get('id_emprendimiento'))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Obtener datos del emprendimiento
    emprendimientos = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/emprendimientos.csv'))
    emprendimiento_tematica = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/emprendimiento_tematica.csv'))
    tematicas = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/tematicas.csv'))
    municipios = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/municipios.csv'))
    seguidores = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/seguidores.csv'))
    publicaciones = pd.read_csv(os.path.join(BASE_DIR, 'DATOS/publicaciones.csv'))
    
    emp = emprendimientos[emprendimientos.id_emprendimiento == id_emprendimiento].iloc[0]
    temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == id_emprendimiento]['id_tematica'].tolist()
    tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in temas]
    municipio = municipios[municipios.id_municipio == emp['id_municipio_origen']]['municipio'].values[0] if emp['id_municipio_origen'] in municipios.id_municipio.values else ''
    seguidores_count = seguidores[seguidores.id_emprendimiento == id_emprendimiento]['cantidad'].values[0] if id_emprendimiento in seguidores.id_emprendimiento.values else 0
    total_likes = publicaciones[publicaciones.id_emprendimiento == id_emprendimiento]['n_likes'].sum() if id_emprendimiento in publicaciones.id_emprendimiento.values else 0
    
    # Obtener recomendaciones
    recommendations = []
    if 'G' in SHARED_DATA and id_emprendimiento in SHARED_DATA['G'].nodes():
        target_idx = np.where(np.array(SHARED_DATA['node_ids']) == id_emprendimiento)[0][0]
        target_embedding = SHARED_DATA['embeddings'][target_idx].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, SHARED_DATA['embeddings'])[0]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if SHARED_DATA['node_ids'][i] != id_emprendimiento][:3]
        
        for idx in sorted_indices:
            rec_id = SHARED_DATA['node_ids'][idx]
            rec_emp = emprendimientos[emprendimientos.id_emprendimiento == rec_id].iloc[0]
            rec_temas = emprendimiento_tematica[emprendimiento_tematica.id_emprendimiento == rec_id]['id_tematica'].tolist()
            rec_tema_names = [tematicas[tematicas.id_tematica == t]['nombre'].values[0] for t in rec_temas]
            recommendations.append({
                'nombre': rec_emp['nombre_emprendimiento'],
                'similitud': float(similarities[idx]),
                'tematicas': rec_tema_names
            })
    
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

def evaluacion(request):
    return render(request, 'simulacion/evaluacion.html')