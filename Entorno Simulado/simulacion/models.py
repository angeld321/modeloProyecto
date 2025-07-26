# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Alcance(models.Model):
    id_alcance = models.AutoField(primary_key=True)
    tipo = models.CharField(max_length=50, db_comment='Local, Regional')

    class Meta:
        managed = False
        db_table = 'alcance'


class Comentario(models.Model):
    id_comentario = models.AutoField(primary_key=True)
    comentario = models.TextField()
    id_publicacion = models.ForeignKey('Publicacion', models.DO_NOTHING, db_column='id_publicacion')

    class Meta:
        managed = False
        db_table = 'comentario'


class Emprendimiento(models.Model):
    id_emprendimiento = models.AutoField(primary_key=True)
    nombre_emprendimiento = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    id_municipio_origen = models.ForeignKey('Municipio', models.DO_NOTHING, db_column='id_municipio_origen')
    id_alcance = models.ForeignKey('Alcance', models.DO_NOTHING, db_column='id_alcance')
    tematicas = models.ManyToManyField('Tematica', through='EmprendimientoTematica')

    class Meta:
        managed = False
        db_table = 'emprendimiento'

class EmprendimientoTematica(models.Model):
    id_emprendimiento = models.ForeignKey(Emprendimiento, models.DO_NOTHING, db_column='id_emprendimiento')
    id_tematica = models.ForeignKey('Tematica', models.DO_NOTHING, db_column='id_tematica')

    class Meta:
        managed = False
        db_table = 'emprendimiento_tematica'
        unique_together = (('id_emprendimiento', 'id_tematica'),)


class Municipio(models.Model):
    id_municipio = models.AutoField(primary_key=True)
    municipio = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'municipio'


class Publicacion(models.Model):
    id_publicacion = models.AutoField(primary_key=True)
    contenido = models.TextField()
    n_likes = models.IntegerField(blank=True, null=True)
    id_emprendimiento = models.ForeignKey(Emprendimiento, models.DO_NOTHING, db_column='id_emprendimiento')

    class Meta:
        managed = False
        db_table = 'publicacion'


class Seguidores(models.Model):
    id_seguidores = models.AutoField(primary_key=True)
    id_emprendimiento = models.ForeignKey(Emprendimiento, models.DO_NOTHING, db_column='id_emprendimiento')
    cantidad = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'seguidores'


class Tematica(models.Model):
    id_tematica = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)

    def __str__(self):
        return self.nombre

    class Meta:
        managed = False
        db_table = 'tematica'
