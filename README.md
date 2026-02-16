1. Start from the original UD trees.
2. Apply pseudo-projective lifting to obtain a projectivized UD representation.
3. Convert the lifted UD trees into the dependency-tree representation $\mathcal{D}$.
4. Train the constituency parser and run prediction $\mathcal{D}^\prime$.
5. Manually correct the predicted structures (intermediate results) $\bar{\mathcal{D}}^\prime$.
6. Train and run the T5 post-editing model for automatic correction $\mathcal{D}^{\prime\prime}$. This includes mapping dependency trees to their linearized representations for T5 training and prediction,
$\mathcal{D} \mapsto \mathcal{L}$ and $\mathcal{D}^\prime \mapsto \mathcal{L}^\prime$, and then mapping the predicted linearized outputs back to dependency trees. Here, $\mathcal{L}$ is the linearization of $\mathcal{D}$.
7. Manually correct the post-edited outputs (final results) $\bar{\mathcal{D}}^{\prime\prime}$.

For evaluation, the outputs are deprojectivized back to the original non-projective UD structures, and scores are reported in the original UD space.

## Quick Start
1. Data preprocessing for upstream model training (projectivization and conllu-to-tree conversion)
```
# main.py
common_preprocessing_pipeline("English", ["train", "dev', "test"], pos="upos")
```
2. Data preprocessing for downstream model training (linearization)
```
# main.py
neural_preprocessing_pipeline("English", ["train", "dev", "test"], is_target=True)
```
3. Data postprocessing (delinearization and replacement with source data [enabled with the option `is_neural=True`], tree-to-conllu conversion, deprojectivization)
```
# main.py
postprocessing_pipeline("English", pos="upos", epochs=100, is_neural=True)
```
4. Upstream model training
```
# train_stanza,ver=autodl.sh
```
5. Upstream model inference
```
# run_stanza,ver=autodl.py
```
5. Downstream mdoel training
```
# train_t5.py
mode = "train"
```
6. Downstream model inference
```
# train_t5.py
mode = "predict"
```
**P.S. I will add argparse later. Sorry for the inconvinence.**
