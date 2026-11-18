# locomo_mc10.cpp

LoCoMo-MC10 multiple-choice scoring for Hy-Memory on the NPU. The full plan, the
prompt format and the results live in `/scratch/michyu/Projects/hy_locomo_eval/NOTES.md`.

## How it works
- Input is `prompts.json` from `Projects/hy_locomo_eval/make_prompts.py`: token ids made
  with the HF tokenizer, so the harness needs no tokenizer or chat template of its own,
  and a CPU reference can read the very same ids.
- 1,986 questions share 10 conversations (11.7K-23K tokens). Per conversation:
  `clear_context()`, `prefill(chat)`, `checkpoint()`; then per question `restore()` and
  `prefill(question)`. hy_mem_npu's checkpoint is only the context length; restore sets it
  back and zeroes the rows the last question wrote (~60 ms). Without this, each question
  would re-read its ~20K-token chat (~2-3 min each, ~200 h in total).
- Scoring reads the logits of the letter tokens A-J (ids 32-41) at the answer position and
  takes the largest, so there is no text to parse. The overall top token is saved too, to
  see whether the model would really answer with a letter.
- `--fresh` skips the reuse and prefills the chat again for every question. Only for
  checking: on conv-30 its logits were bit-identical to the reuse path.
- The context length is passed straight to `load_model` (default 32768 here); the model
  list's 16384 default would not fit the longest prompt (23,257 tokens).
