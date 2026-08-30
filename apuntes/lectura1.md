

# Kelleher - Cap 1

- **Inductive Bias** (restriction & preference bias): Mínimo conjunto de asunciones tal que 

$$(\forall x \in X)[(B \land D_c \land x) \vdash L(x,D_c)]$$

- **Goldilocks Model**. Modelo con buen balance (ni overfit ni underfit)

### Cross Industry Standard Process For Data Mining (CRISP-DM):
1. Business understanding
2. Data understanding
3. Data preparation
4. Modelling
5. Evaluation
6. Deployment

- **Predictive Data Analytics**: Construcción y uso de modelos para hacer predicciones basadas en patrones extraidos de data histórica.

- **Supervised Learning Model**: Técnicas para aprender un modelo de la relación del conjunto de descripciones de atributos y targets, basado en instancias históricas.

--- 

# Mitchell - Cap 1

- **Concept Learning:** Inferencia de una función de clasificación a partir de datos de entrenamiento y su output. Se lo puede pensar como una busqueda de la hipótesis que mejor estima una funcion objetivo.

- **Inductive Learning Hypothesis**: Un modelo que performa bien con los datos de entrenamiento, si estos son muchos, va a performar bien con datos nuevos.

- **Orden de hipótesis**: Se define una relación de orden entre hipótesis
$$h_j \geq_g h_k \iff (\forall x \in X)[(h_k(x)=1) \rightarrow (h_j(x)=1)]$$ 

- **Consistencia:** $\text{consistent(h,D)}= (\forall \langle x, c(x) \rangle \in D) (h(x)=c(x))$

### FindS:
Algoritmo que busca la hipótesis más específica consistente con los ejemplos positivos (ignora los negativos):
1. Inicializar $h$ con la hipótesis más específica posible ($h = \langle \emptyset, \dots, \emptyset \rangle$)
2. Para cada ejemplo de entrenamiento positivo $x$:
   - Para cada atributo $a_i$ de $h$: si $x$ no satisface $a_i$, reemplazarlo por la siguiente restricción más general que sí satisface $x$
3. Devolver $h$

- Asume que existe una hipótesis consistente en $H$ ($c \in H$) y que los datos no tienen ruido.
- No detecta inconsistencias ni indica cuántas hipótesis alternativas existen.

## Candidate Elimination Algorithm: 
Mantiene y actualiza incrementalmente el **Version Space** completo (no solo una hipótesis), usando tanto ejemplos positivos como negativos.

- **Version Space** ($VS_{H,D}$): Subconjunto de hipótesis de $H$ consistentes con los ejemplos de entrenamiento $D$.
$$VS_{H,D} = \{h \in H \mid \text{consistent}(h,D)\}$$
Se representa de forma compacta mediante sus dos fronteras (S y G), evitando enumerar todas las hipótesis.

- **Specific Boundary (S)**: Las hipótesis *más específicas* (menos generales) que siguen siendo consistentes con $D$ — no existe ninguna consistente que sea aún más específica.
$$S = \{s \in H \mid \text{consistent}(s,D) \land (\nexists s' \in H)[(s \geq_g s') \land s \neq s' \land \text{consistent}(s',D)]\}$$
  Ante un ejemplo **positivo** no cubierto por algún $s \in S$, se generaliza $s$ lo mínimo necesario para cubrirlo; ante un ejemplo **negativo** cubierto por algún $s \in S$, esa hipótesis se elimina (es demasiado general, "sobre-cubre").

- **General Boundary (G)**: Las hipótesis *más generales* que siguen siendo consistentes con $D$ — no existe ninguna consistente que sea aún más general.
$$G = \{g \in H \mid \text{consistent}(g,D) \land (\nexists g' \in H)[(g' \geq_g g) \land g \neq g' \land \text{consistent}(g',D)]\}$$
  Ante un ejemplo **negativo** cubierto por algún $g \in G$, se especializa $g$ lo mínimo necesario para excluirlo; ante un ejemplo **positivo** no cubierto por algún $g \in G$, esa hipótesis se elimina (es demasiado específica, "sub-cubre").

**Diferencia clave**: $S$ arranca en la hipótesis más específica posible ($\emptyset$) y se va generalizando *solo lo justo* para no perderse ningún positivo — es la frontera "conservadora" (mínimo riesgo de falsos positivos). $G$ arranca en la hipótesis más general posible ($?$) y se va especializando *solo lo justo* para no aceptar ningún negativo — es la frontera "permisiva" (mínimo riesgo de excluir positivos futuros). El Version Space es exactamente el conjunto de hipótesis intermedias entre ambas fronteras:
$$VS_{H,D} = \{h \in H \mid \exists s \in S, \exists g \in G : g \geq_g h \geq_g s\}$$

- **Query Strategy**: Estrategia para elegir el próximo ejemplo de entrenamiento a etiquetar de forma óptima: se busca un ejemplo sobre el cual las hipótesis del Version Space discrepen en su clasificación, de modo que la respuesta (positiva o negativa) elimine aproximadamente la mitad de las hipótesis restantes, minimizando así la cantidad de ejemplos necesarios para converger a una única hipótesis.

### Inducción a Deducción
- Inducción: $(D_c \land x_i) \succ L(x_i, D_c)$
- Deducción: $(B \land D_c \land x_i) \vdash L(x_i, D_c)$

---