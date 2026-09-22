## Overview
This project aims to convert non-projective sentences into pseudo-projective tree representations, making them compatible with parsing models that have linear inference complexity. Traditionally, parsing non-projective sentences requires algorithms with higher computational complexity, such as the $(O(n^3))$ MST parser. With the proposed data representation, however, these sentences can be parsed in $(O(n))$ time. The main innovation is the observation that a sentence can be converted into a tree representation only if it is projective or pseudo-projective. Therefore, projectivization must take place before the conversion to trees (constituentization).

## Key feature
We use the Stanza Constituency Parser for upstream predictions. One challenge of using a constituency parser for dependency parsing is that the resulting subtrees may be headless or contain multiple heads. To enforce structural validity, we designed a BFS-plus-recursion algorithm to select a reasonable head for headless or multi-head subtrees (tokens).

## Architecture
### Flowcharts of the architecture
<img width="300" alt="upsteam-preprocessing" src="https://github.com/user-attachments/assets/d000b637-b775-4ab3-b16d-f3c5951c6447" /> \
Figure 1. Upstream preprocessing \
<img width="300" alt="upstream-postprocessing" src="https://github.com/user-attachments/assets/f0571747-5fcc-436a-ad7e-67ccacfc83af" /> \
Figure 2. Upstream postprocessing, Downstream pre- and postprocessing \
<img width="300" alt="custom-model-processing" src="https://github.com/user-attachments/assets/dc6ba552-dbd7-4cd8-b8ee-c6cdc144dcbd" /> \
Figure 3. Alternative Custom model pre- and postprocessing

### Description of the workflow
1. Start from the original UD trees.
2. Apply pseudo-projective lifting to obtain a projectivized UD representation.
3. Convert the lifted UD trees into the dependency-tree representation $\mathcal{D}$.
4. Train the constituency parser and run prediction $\mathcal{D}^\prime$.
5. Manually correct the predicted structures (intermediate results) $\bar{\mathcal{D}}^\prime$.
6. Train and run the T5 post-editing model for automatic correction $\mathcal{D}^{\prime\prime}$. This includes mapping dependency trees to their linearized representations for T5 training and prediction,
$\mathcal{D} \mapsto \mathcal{L}$ and $\mathcal{D}^\prime \mapsto \mathcal{L}^\prime$, and then mapping the predicted linearized outputs back to dependency trees. Here, $\mathcal{L}$ is the linearization of $\mathcal{D}$.
7. Manually correct the post-edited outputs (final results) $\bar{\mathcal{D}}^{\prime\prime}$.

For evaluation, the outputs are deprojectivized back to the original non-projective UD structures, and scores are reported in the original UD space.

## Data Formats
### [CoNLL-U] Raw CoNLL-U
```
# sent_id = tlg0008.tlg001.perseus-grc1.13.tb.xml@1207
# text = καὶ γὰρ ὁ νομοθέτης Σόλων ἔφη·
1	καὶ	καί	ADV	d--------	_	5	advmod	_	_
2	γὰρ	γάρ	ADV	d--------	_	6	advmod	_	_
3	ὁ	ὁ	DET	l-s---mn-	Case=Nom|Gender=Masc|Number=Sing	5	det	_	_
4	νομοθέτης	νομοθέτης	NOUN	n-s---mn-	Case=Nom|Gender=Masc|Number=Sing	5	nmod	_	_
5	Σόλων	Σόλων	NOUN	n-s---mn-	Case=Nom|Gender=Masc|Number=Sing	6	nsubj	_	_
6	ἔφη	φημί	VERB	v3siia---	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act	0	root	_	SpaceAfter=No
7	·	·	PUNCT	u--------	_	6	punct	_	_
```


### [CoNLL-U] Pseudo-projective CoNLL-U
We experimented with the Head+Path, Head, and Path labeling schemes proposed by Nivre and Nilsson (2005). Unlike their experiments, which used deterministic models, we use neural models. Head+Path retains the most information and therefore performs best with deterministic models, whereas Path has the smallest label set and performs best with neural models. Based on these results, we selected Path as our labeling scheme. As shown in the projectivized representations, upward arrows mark the lifted arcs, while downward arrows mark the original ancestors that the lifted arcs pass through during lifting.
```
# sent_id = tlg0008.tlg001.perseus-grc1.13.tb.xml@1207
# text = καὶ γὰρ ὁ νομοθέτης Σόλων ἔφη·
1	καὶ	καί	ADV	d--------	_	6	advmod↑	_	_
2	γὰρ	γάρ	ADV	d--------	_	6	advmod	_	_
3	ὁ	ὁ	DET	l-s---mn-	Case=Nom|Gender=Masc|Number=Sing	5	det	_	_
4	νομοθέτης	νομοθέτης	NOUN	n-s---mn-	Case=Nom|Gender=Masc|Number=Sing	5	nmod	_	_
5	Σόλων	Σόλων	NOUN	n-s---mn-	Case=Nom|Gender=Masc|Number=Sing	6	nsubj↓	_	_
6	ἔφη	φημί	VERB	v3siia---	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act	0	root	_	SpaceAfter=No
7	·	·	PUNCT	u--------	_	6	punct	_	_
```

## [Penn Tree Bank] Pseudo-projective, Constituentized Trees in .mrg file
Tree representations were only available for constituent structures. We extend this representation to dependency structures by wrapping dependency relations around the original constituents, allowing dependency relations to be represented in a tree format.
```
(TOP (root (advmod↑ (ADV καὶ)) (advmod (ADV γὰρ)) (nsubj↓ (det (DET ὁ)) (nmod (NOUN νομοθέτης)) (NOUN Σόλων)) (VERB ἔφη) (punct (PUNCT ·))))
```

## [Google] Pseudo-projective, Linearized Trees .txt file
We follow the linearization method proposed by Google. The sentence is delexicalized, with POS tags representing the actual tokens in the sentence. Dependency relations are represented using balanced opening and closing brackets, such as `(nsubj` and `)nsubj`. Each POS tag and dependency bracket is treated as an individual token in the linearized representation.
```
(TOP (root (advmod-up ADV )advmod-up (advmod ADV )advmod (nsubj-down (det DET )det (nmod NOUN )nmod NOUN )nsubj-down VERB (punct PUNCT )punct )root )TOP
```


## [Custom] Structured Tokens in .json file
The structured token format was developed based on the linearized representation. We assign left and right features to each actual token, represented by its POS tag. The left feature consists of the left brackets before the token, while the right feature consists of the right brackets after the token, stopping when the first left bracket is encountered. We also introduce an overlap variable to capture structural overlap between adjacent tokens. For example, if overlap = 3, three brackets in the left feature of the following token overlap with the right feature of the previous token.
```
 [
        {
          "token": "ADV",
          "left": [
            "(TOP",
            "(root",
            "(advmod-up"
          ],
          "right": [
            ")advmod-up"
          ]
        },
        {
          "token": "ADV",
          "left": [
            "(advmod"
          ],
          "right": [
            ")advmod"
          ]
        },
        {
          "token": "DET",
          "left": [
            "(nsubj-down",
            "(det"
          ],
          "right": [
            ")det"
          ]
        },
        {
          "token": "NOUN",
          "left": [
            "(nmod"
          ],
          "right": [
            ")nmod"
          ]
        },
        {
          "token": "NOUN",
          "left": [],
          "right": [
            ")nsubj-down"
          ]
        },
        {
          "token": "VERB",
          "left": [],
          "right": []
        },
        {
          "token": "PUNCT",
          "left": [
            "(punct"
          ],
          "right": [
            ")punct",
            ")root",
            ")TOP"
          ]
        }
      ],
```



## Quick Start
1. Data preprocessing for upstream model training (projectivization and conllu-to-tree conversion)
```
scripts/main.py
common_preprocessing_pipeline(lang="Ancient_Greek", split="train", pos="UPOS", head=None, path=None, labels_aligned=False)
```
2. Data preprocessing for downstream model training (linearization, linearized-to-structured tokens conversion)
```
scripts/main.py
downstream_preprocessing_pipeline(lang="Ancient_Greek", split="train", pos="XPOS", epochs=20, overlap=0, is_target=False)
```
3. Data postprocessing (structured tokens-to-linearized conversion, delinearization, tree-to-conllu conversion, deprojectivization)
```
scripts/main.py
postprocessing_pipeline(lang="Ancient_Greek", pos="UPOS", epochs=100, subfolder="label_experiments", head=head, path=path)
```
4. Upstream model training
```
src/models/stanza/train_stanza.sh
```
5. Upstream model inference
```
src/models/stanza/run_stanza.py
```
