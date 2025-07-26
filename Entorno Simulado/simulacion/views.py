from django.shortcuts import render
from django.http import HttpResponse
from .models import Emprendimiento, Municipio, Tematica, Alcance


def lista_emprendimientos(request):
    # Obtener filtros desde los parámetros GET
    municipio_id = request.GET.get('municipio')
    alcance_id = request.GET.get('alcance')
    tematica_id = request.GET.get('tematica')

    # Consulta base con optimización
    emprendimientos = Emprendimiento.objects.all().select_related('id_municipio_origen', 'id_alcance').prefetch_related('tematicas')

    # Aplicar filtros si existen
    if municipio_id:
        emprendimientos = emprendimientos.filter(id_municipio_origen__id_municipio=municipio_id)
    if alcance_id:
        emprendimientos = emprendimientos.filter(id_alcance__id_alcance=alcance_id)
    if tematica_id:
        emprendimientos = emprendimientos.filter(tematicas__id_tematica=tematica_id)

    # Obtener opciones para los filtros
    municipios = Municipio.objects.all()
    alcances = Alcance.objects.all()
    tematicas = Tematica.objects.all()

    context = {
        'emprendimientos': emprendimientos,
        'municipios': municipios,
        'alcances': alcances,
        'tematicas': tematicas,
    }
    return render(request, 'simulacion/emprendimientos.html', context)
# Create your views here.
def hello(requests):
    return HttpResponse("<h1>Hello World</h1>")


def about(request):
    return HttpResponse("<h1>About</h1>")