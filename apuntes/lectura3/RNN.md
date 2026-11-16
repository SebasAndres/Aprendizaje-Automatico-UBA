## RNN

> En toda RN la función de activación final está determinada por el problema.
> Si las funciones de activacion son la identidad --> Regresión Lineal (g'(g(X)) == W'W x = Z x).

---

## Problema predicción de secuencias

Por ejemplo: Clasificación de documentos.

### Posibles soluciones:

#### Representaciones clásicas

- BoW. 
    Ejemplo: (texto="Hola como estas" --> v=<0,0,1,0,1,1>, donde v_i = #v_i in texto)
    Contras:
        Vectores muy esparsos.
        No mantenes la cronología de la palabra (la palabra sin su contexto pierde información)

- Bag-Of-N-Grams
    Mantener n-palabras consecutivas y contar sus apariciones.
    Ejemplo: (3-gram="no-me-gusto" in text="no me gustó mucho la pelicula")

#### Nuevas representaciones

- Word-Embeddings: 
    La idea es que dos vectores están cerca si su significado es similar.
    Existen los contextuales (los + usados hoy en día) y no contextuales (FastText, GloVe, Word2Vec, BERT).

---
## Tipos de modelos de prediccion de secuencias

| Input | Output | Ejemplo |
| ---|---|---|
| 1 | Muchos | Recibe una imagen y responder "Veo un perro" |
| Muchos | 1 | Predecir sentimiento de una oración | 
| Muchos | Muchos | Predecir predio del dolar en un día (en distintos horarios) |
| Muchos | Muchos (desfasado)| Predecir siguientes tokens dados unos tokens iniciales / Traducción de texto (no hay alineación de palabra-palabra entre idiomas) |  

Vamos a ver el caso "Muchos a Muchos" ya que sirve para generalizar todos estos casos. 

### Muchos-a-Muchos

Sea $y^{(i)}, o^{(i)} \in \mathbb{R}^T$ los valores reales y predicción para cada instante $t < T$.

Si defino $J_{X,y} = \frac{1}{N} \sum_i^n L(o^{(i)}, y^{(i)})$ con $L(\{ o_1^(k),...,o_T^{(k)} \}, \{ y_1^(k),...,y_T^{(k)} \})$

---

