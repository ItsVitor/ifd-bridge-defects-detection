# Experiment Workflow

Every model experiment follows `docs/experiments.md` and is recorded with `docs/experiment_template.md`. Records are written in Brazilian Portuguese.

## Rules

- **Write the rationale before the run.** Before proposing or running an experiment, create `outputs/<model>/E{NNN}_{slug}/` with a frozen `config.yaml` and a `README.md` filling sections 1–6: motivation with cited evidence, a refutable hypothesis, the change vs. the parent experiment, predictions, an adoption criterion, and the origin of every parameter. Present it to the user and do not run until they approve.
- **One change per experiment** relative to its parent. If more than one is unavoidable, state why and which effects are confounded.
- **Declare the origin of each choice** (Medido, Físico, Convenção, Herdado, Pedido, Arbitrário). Never invent a justification for an arbitrary value; label it Arbitrário.
- **Keep the unsupervised premise.** Motivation and adoption criteria rely only on healthy data or physical reasoning; test metrics are reported, never used to adopt a change.
- **Never overwrite or reuse an ID.** A rerun gets a new ID. Failed and abandoned experiments stay recorded.
- **After the run,** fill sections 7–10, compare each prediction with the outcome, record the decision, and update `outputs/<model>/README.md`, including the count of variants evaluated on the test folds.
- **Do not rewrite sections 1–6 after the run.** Corrections go to "Notas posteriores" with a date.
