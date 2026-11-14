# Liste de Modèles 
Pour visualiser le fichier markdown:
sous Windows : Ctrl+K puis V
sous Mac : Cmd+K puis V 
## Transformers Traduction
type Sequence to Sequence

### MarianMT 
développé par le Language Technology Research Group de l'Université d'Helsinki / Helsinki-NLP
All MarianMT models are transformer encoder-decoders with 6 layers in each component, use static sinusoidal positional embeddings, don’t have a layernorm embedding.

https://huggingface.co/docs/transformers/model_doc/marian
https://huggingface.co/transformers/v3.5.1/model_doc/marian.html

### mBART 
(Multilingual BART) développé par MetaAI
https://huggingface.co/docs/transformers/model_doc/mbart	

## Transformers Q&A

### type Encodeur

#### BERT 
(Bidirectional Encoder Representations from Transformers) développé par Google en 2018
	Encodeur (extraction de réponse)
https://huggingface.co/docs/transformers/model_doc/bert

#### ELECTRA 
développé par des chercheurs de l 'Université Stanford et de Google Brain (Google Research)

##### Principe
modifie l’objectif de pré-entraînement des modèles de langage masqués traditionnels comme BERT. Au lieu de simplement masquer des tokens et de demander au modèle de les prédire, ELECTRA entraîne deux modèles : un générateur et un discriminateur. Le générateur remplace certains tokens par des alternatives plausibles, et le discriminateur (le modèle que vous utiliserez réellement) apprend à détecter quels tokens sont originaux et lesquels ont été remplacés. Cette approche d’entraînement est très efficace et passe à l’échelle sur des modèles plus grands, tout en utilisant considérablement moins de calcul.
Cette approche est très efficace parce que ELECTRA apprend de chaque token de l’entrée, et pas uniquement des tokens masqués. C’est pourquoi même les petits modèles ELECTRA peuvent égaler, voire surpasser, des modèles bien plus grands, tout en consommant beaucoup moins de ressources de calcul
https://huggingface.co/docs/transformers/model_doc/electra

### type encodeur/décodeur

#### T5 
(Text-to-Text Transfer Transformer) développé par Google AI
https://huggingface.co/docs/transformers/model_doc/t5

## LMM (2 tâches)

### DeepSeek 
fondé par Liang Wenfeng
https://huggingface.co/deepseek-ai/deepseek-llm-67b-base

### Qwen 3.8 
développé par Alibaba
https://huggingface.co/Qwen/Qwen3.8-Flash-Next

### Translate Gemma 
(Google DeepMind)
https://huggingface.co/google/translategemma-4b-it

### Phi 4 
(Microsoft)
https://huggingface.co/microsoft/Phi-4-mini-instruct

