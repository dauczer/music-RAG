# Evaluation

This project deliberately uses a small, inspectable evaluation instead of a RAG
evaluation platform.

## Retrieval

`questions.jsonl` contains 18 hand-written cases. Answerable cases name one or
more acceptable song titles for each artist. The runner checks the metadata of
the top results and reports Hit@5 plus MRR@5:

```bash
python -m evals.run --output evals/runs/latest.json
```

The command uses the same Hugging Face query embedding as production and needs
`HF_TOKEN`. Run it against a staged index with:

```bash
CHROMA_DB_PATH=chroma_db_v2 python -m evals.run
```

The initial thematic labels are deliberately small. Read the candidate songs
before changing them; do not add a title merely because the current retriever
returns it. Keep at least half of the evaluation questions out of the portfolio's
visible suggestions.

Initial release targets:

- title lookup: 4/4 hits in the top five;
- all answerable retrieval checks: at least 12/15 checks, with both artist checks
  passing for a comparison;
- MRR@5 of at least 0.60.

Because comparisons create one retrieval check per artist, the JSON report's
`retrieval_checks` count is higher than its number of questions.

## Answer review

Review eight representative generated answers manually. Score each criterion
from 0 to 2:

1. Every substantive claim is supported by a returned source.
2. The response actually answers the question.
3. Displayed source titles match the retrieved metadata.
4. Uncertainty is explicit when evidence is incomplete.

A response passes at 7/8 or better, but criteria 1 and 3 must both score 2.
Also run all cases where `answerable` is false and record whether the API returns
`status: insufficient`. The initial target is at least 5/6 correct refusals and
no more than one false refusal among answerable questions.

Do not use an LLM judge as the sole acceptance signal. If one is added later,
keep the source-based metrics and the manual sample.
