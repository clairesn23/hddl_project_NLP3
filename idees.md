# Architecture/Organisation

## Création de deux notebooks :
1 -> Question et Réponsé
2 -> Traduction

Dans chaque notebook, métriques communes 

## Ajout notebook supplémentaire 
Obtenir un jeu de données de test qui est semantic preserving : LMM as judge, Cosine similarity 

# Quel modèle/métrique/dataset prenons nous ?

dataset : un par tâche 

Q&A: 
https://github.com/clairesn23/hddl_project_NLP3.git

Perturber : 
https://github.com/IBM/perturber

modèle de Q&A
- transformer 1: https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad
modèle traduction
1 tranformer et 1 LLM dont 1 LLM commun à la traduction et Question/Réponse

2 transformers (trad), 2 transformers (Q&A), 2 LMM (les2)
pour la semaine du 18: recherches doivent être finies
pour la semaine du 25: structuration des notebooks avancée (eventuellement premier test simple)