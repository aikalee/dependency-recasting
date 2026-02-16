1. Start from the original UD trees.
2. Apply pseudo-projective lifting to obtain a projectivized UD representation.
3. Convert the lifted UD trees into the dependency-tree representation $\mathcal{D}$.
4. Train the constituency parser and run prediction $\mathcal{D}^\prime$.
5. Manually correct the predicted structures (intermediate results) $\bar{\mathcal{D}}^\prime$.
6. Train and run the T5 post-editing model for automatic correction $\mathcal{D}^{\prime\prime}$. This includes mapping dependency trees to their linearized representations for T5 training and prediction,
$\mathcal{D} \mapsto \mathcal{L}$ and $\mathcal{D}^\prime \mapsto \mathcal{L}^\prime$, and then mapping the predicted linearized outputs back to dependency trees. Here, $\mathcal{L}$ is the linearization of $\mathcal{D}$.
7. Manually correct the post-edited outputs (final results) $\bar{\mathcal{D}}^{\prime\prime}$.

For evaluation, the outputs are deprojectivized back to the original non-projective UD structures, and scores are reported in the original UD space.

## Repository Structure
```
main.py
src/
  pathgen.py
  common/
  preprocessing/
  postprocessing/

```

