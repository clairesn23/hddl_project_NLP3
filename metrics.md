# Comment mesurer la robustesse de modèles NLP ?

## 1. Définition de la robustesse

On définit la robustesse comme la capacité d'un modèle à préserver la sémantique des sorties face à des changements dit "mineurs" de l'entrée.

Posons $(x,x')$ dans $I$, ensemble d'entrée du modèle, supposés identiques sémantiquement, et $f : I \to U$, tel que $f(x) = y$ représente la sortie du modèle.
On cherche alors à évaluer :

$$R(x,x') = \left\| f(x)-f(x') \right\| =  \left\| y-y' \right\|$$

Avec $R$, robustesse qui équivaut à une "mesure de distance sémantique".

Il nous faut alors définir comment mesurer une distance sémantique entre 2 éléments textuels.

## 2. Première intuition

Une mesure de distance se fait généralement sur un espace vectoriel commun.
Une première manière d'évaluer serait de passer par un encodeur pour transformer $y$ et $y'$ en vecteurs de dimension $N$ dans un espace latent de taille $N$. *(Principe d'embedding)*

La distance pourrait ensuite se mesurer par cosine similarity, soit:

$$ d(A,B) = \frac{A \cdot B}{\|A\| \|B\|} $$

Avec, $A$ et $B$, les vecteurs associés à $y$ et $y'$ respectivement.

**Question :** Quel encodeur utiliser ? Toute transformation vectorielle peut influencer sur la robustesse en elle-même selon la tokenization des mots.

**Solutions possibles :**

* sentence-transformers/multilingual-e5-small : modèles d'embedding
* preprocessing + TF-IDF (Term Frequency – Inverse Document Frequency) : https://github.com/mifthulyn07/tfidf-and-cosine-similarity


Un score utilisant un LLM-as-judge peut également être réalisé, mais il s'agit encore d'un résultat à l'appréciation du modèle choisi.

## 3. Une définition appliquée à des modèles de traduction

On peut également définir la robustesse comme la consistence du comportement du modèle. C'est ce qui est définit dans l'article *"Measuring Robustness for NLP" (Y. Yu et al., 2022)*.

Cette définition amène à mesurer la variance du score de performance entre deux sorties de modèle NLP en tâche de traduction, plus exactement, la variance du score **BLEU (Bilingual Evaluation Understudy)**.

### 3.1. Métriques standards pour l'évaluation de modèles NLP

*Référence - https://rexresearch1.com/LargeLanguageModelsLibrary/HandsOn%20Large%20Language%20Models%20LanguageAlammar.pdf*

"Common word-level metrics include perplexity, ROUGE,
BLEU, and BERTScore."

D'après le livre "Hands-On Large Language Models" par Jay Alammar, ces quatres métriques sont les scores les plus communs pour évaluer des modèles de génération de langage, à l'échelle des mots et tokens.

De manière simple, chaque méthode évalue :
* la performance de prédiction de texte et de tokens (perplexity)
* la performance de génération de sommaires (ROUGE)
* la performance de traduction de texte (BLEU)
* la performance basée sur la représentation contextuelle (BERTScore - cf. partie 2) 

### 3.2. Score BLEU : Définition et implémentation

Ici, nous appliquerons le score BLEU en cohérence avec la tâche de traduction. *(https://aclanthology.org/P02-1040.pdf, "BLEU: a Method for Automatic Evaluation of Machine Translation", K. Papineni et al., 2002)*

Il s'agit d'un score qui permet d'évaluer la performance de traduction en comparaison à une traduction de référence, générée par l'homme. Il compare les *n-grammes* (les séquences de n-mots consécutifs) et calcule la précision en comparant ces séquences à celles de références. Ce score est également pénalisé vis-à-vis des traductions plus courtes.

On obtient la version simplifiée de la formule *(https://codelabsacademy.com/fr/blog/understanding-bleu-score-in-nlp-evaluating-translation-quality/)* :


$$\text{BLEU} = \text{BP}*\exp{(\frac{1}{n}\sum_{i=1}^{n}{\log{p_i}})}$$

Avec $n$, l'ordre maximal des n-grammes, $p_i$, la précision à l'ordre $i$ et $\text{BP}$, la pénalité appliquée aux traductions plus courtes que celle de référence.

On a également :

$$p_i = \frac{\text{nombre de i-grammes du texte généré}}{\text{nombre de i-grammes du texte de référence}}$$

$$ \text{BP} = \left\{
\begin{array}{l}
1, \text{ si c > r} \\
\exp{(1 - \frac{r}{c})}, \text{ sinon }
\end{array}
\right. $$

Avec $c$, longueur du texte généré, et $r$, longueur du texte de référence.

**Implémentation en Python :**
* https://koshurai.medium.com/demystifying-the-blue-score-a-python-implementation-for-evaluating-language-quality-c44db88738e0
* https://www.geeksforgeeks.org/nlp/nlp-bleu-score-for-evaluating-neural-machine-translation-python/#bleu-score-implementation-in-python

### 3.3. Mesure de la variance du score BLEU (méthode probabiliste)

Pour rendre la variance interprétable quelque soit le dataset utilisé, l'article de Y. Yu présente le coefficient de variation, ou variance stable qui est défini par la variance pondérée par la moyenne du jeu de données.

L'article définit alors qu'un NLP est dit $(\epsilon, \gamma)-\text{robust}$, si pour tout jeu de test tiré d'une distribution $D$, l'erreur de prédiction $X$ est centré autout de l'erreur moyenne $\mu$, bornée par le paramètre $\epsilon$ à la probabilité $1-\frac{\sigma^2}{\epsilon^2}\times\gamma$.
On écrit :

$$Pr[|X-\mu|<\epsilon ] = 1-\frac{\sigma^2}{\epsilon^2}\times\gamma$$

Avec $X$, qui correspond ici à la variable aléatoire représentant la performance mesurée par le score BLEU du modèle.

Pour la mesure de la robustesse, l'interprétation dépend de la valeur de $\gamma \in [0,1]$. Si $\gamma = 1$, le modèle n'est pas du tout robuste. A l'inverse, quand $\gamma \rightarrow 0$, le modèle est plus proche du meilleur scénario.

Cette relation vient de l'inégalité de Chebyshev et est détaillée dans l'article. Elle permet de représenter la robustesse de manière probabiliste.*(Y. Yu et al., 2022)*

L'article présente également 3 métriques pour mesurer la variance du score BLEU selon la définition de la robustesse et les stabiliste par des stratégies de bootstrap et de bagging. L'implémentation officielle n'est pas disponible de manière publique.


## 4. Des métriques spécificiques aux modèles Q&A

Les métriques communes utilisées pour les modèles Q&A sont le score F1 et le score EM (Exact Match).

Le score EM associe directement une réponse de référence à la réponse générée. Il vaut 1 si les deux réponses se correspondent, 0 sinon.

Le score F1 respecte les définitions suivantes :

$$\text{F1-score} = \frac{2 \times \text{Recall}\times\text{Precision} }{\text{Recall} + \text{Precision}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}} = \frac{\text{nombre de tokens communs}}{\text{nombre de tokens générés}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}} = \frac{\text{nombre de tokens communs}}{\text{nombre de tokens de référence}}$$


**Exemple d'implémentation Python :**

https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Evaluating-a-model-on-the-SQuAD2.0-dev-set-with-HF