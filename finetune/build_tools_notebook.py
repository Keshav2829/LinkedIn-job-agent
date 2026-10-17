"""One-off builder for finetune/train_gemma4_e4b_qlora_tools.ipynb -- run
once to (re)generate the notebook file from this Python source, so the
cells are easy to review/diff as code rather than as raw nbformat JSON.
Not imported by anything; delete after the notebook looks right, or keep
it around as the source of truth for future edits (regenerate rather than
hand-editing the .ipynb's JSON).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "train_gemma4_e4b_qlora_tools.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _lines(text)}


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    md(
"""# gemma-4-E4B QLoRA fine-tune — orchestrator tool-use/tool-selection

Trains the **orchestrator** agent (`orchestrator/agent.py`, the shell-driven
loop behind `python -m orchestrator run ...`), not the browsing/screenshot
agent `finetune/train_gemma4_e4b_qlora.ipynb` trains. Different corpus,
different target output format:

| | this notebook | `train_gemma4_e4b_qlora.ipynb` |
|---|---|---|
| corpus | `orchestrator/finetune.py`'s `export-tool-data` (real orchestrator sessions) | `annotation_suite`-reviewed harvested browsing steps |
| target format | a ` ```action ` fence: `{"tool": "run_shell", "input": {...}}` / `{"final": ...}` / `{"ask": ...}` | `<thinking>...</thinking>` + `<action kind="...">...</action>` |
| base model | `unsloth/gemma-4-E4B-it` | same |

Both target the same base model (`orchestrator/config.json`'s `model` field
currently points LM Studio at `gemma-4-e4b-linkedin`, served from the
*other* notebook's GGUF) — but there's no retained LoRA adapter artifact
for that checkpoint in this repo to continue from, so this trains a fresh,
**separate** LoRA pass from the base model. Give the result its own name in
LM Studio (e.g. `gemma-4-e4b-linkedin-tools`) rather than overwriting the
browsing checkpoint; see cell 10.

**Data**: `python -m orchestrator finetune export-tool-data` derives two
datasets from exported orchestrator sessions — `tool_use.jsonl` (one
example per real tool call) and `tool_select.jsonl` (every decision point:
every tool call, **plus** `final`/`ask` — a strict superset that also
teaches *when not* to call a tool). **Use `tool_select.jsonl`** — training
on both in the same run would just re-teach `tool_select`'s `tool`-kind
rows a second time, since every `tool_use.jsonl` row already appears inside
`tool_select.jsonl`.

**Why `--max-context-chars` matters here**: unlike the browsing corpus
(fixed 3-turn examples), an orchestrator session can run 100+ turns, and
each example's `messages` is normally the *entire* prefix up to that one
decision — a late decision in a long session carries the whole history.
Measured on a real export: median ~29k tokens, max ~66k (rough 4
chars/token estimate) — far past a free-tier GPU. The export command's
`--max-context-chars` truncates each example instead: always keeps the
system prompt, the original task, and the decision turn itself (the
training target), then fills backward with as much recent context as fits
— the same tradeoff a sliding context window makes at real inference time.
Regenerate (or re-use the already-exported files in this repo) with:

```bash
python -m orchestrator finetune export-tool-data \\
  --in orchestrator/data/finetune/export3.jsonl \\
  --val-frac 0.2 --max-context-chars 24000
```

On the export this notebook was written against: 372 train / 118 val
decision points after truncation, of which **131 train / 74 val carry an
embedded screenshot** (the orchestrator's `screenshot` tool really does
attach real images to the conversation, same as the browsing corpus).

**This round trains on images too, not just text.** The first version of
this notebook dropped every screenshot-bearing row, the same workaround
`train_gemma4_e4b_qlora.ipynb` still uses, because of
[unslothai/unsloth#1436](https://github.com/unslothai/unsloth/issues/1436)
— `UnslothVisionDataCollator` erroring on a dataset that mixes (or
entirely lacks) image-bearing rows. That issue is now **closed**: a
maintainer confirmed text-only/mixed training is supported as of a recent
`unsloth`/`unsloth_zoo` release, and cell 1 already installs unpinned
`unsloth` (latest). So this round keeps every row — 372 train / 118 val,
images included — and uses the real vision training path (`UnslothVisionDataCollator`,
cell 8) instead of flattening everything to plain text and throwing the
images away.

**This is unverified against this exact `unsloth` + `transformers` +
`gemma-4` combination** — I can't run a GPU session myself to confirm it
end to end. If cell 4 or cell 6 errors on the image/text mix specifically
(not just an unrelated API-rename error — see the last paragraph below),
the previous, verified-working version of this notebook (drop images,
train text-only via plain `SFTConfig(dataset_text_field="text")`) is
recoverable from git history (`git log -- finetune/train_gemma4_e4b_qlora_tools.ipynb`)
rather than something to rebuild from scratch.

**Loss is still plain full-sequence, not assistant-turn-only masking.**
Most of each transcript's token budget is tool-output text (`run_shell`
stdout, page text) the model isn't meant to reproduce, so some gradient
signal is spent on unpredictable text rather than the actual decisions.
Worth revisiting (TRL's `SFTConfig(assistant_only_loss=True)`, if your TRL
version and chat template support it) once this pass proves the vision
data pipeline itself works end to end — changing two things at once would
make a failure harder to diagnose.

**Before running**: generate (or confirm) `tool_select.jsonl` and
`tool_select.val.jsonl` locally per the command above, then make them
available to cell 3 — a Kaggle Dataset (Notebook > Add Input) or Colab's
upload dialog, whichever this session is running on.

**If a cell errors on an API mismatch**: `gemma-4` is very recent and
Unsloth's API for it may have moved since this notebook was written (Oct
2026). Check https://unsloth.ai/docs/basics/vision-fine-tuning and
https://unsloth.ai/docs/models/gemma-4/train against the cells below — the
fix is almost always a renamed argument, not a change in approach."""),
    md("## 1. Install"),
    code(
"""%%capture
import os

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2"""),
    md(
"""## 2. Load base model (4-bit)

Same base model as the browsing notebook — `FastVisionModel`, not
`FastLanguageModel`, so the vision tower survives the merge/export step
even though this round's gradients never touch it (`finetune_vision_layers=False`,
cell 5)."""),
    code(
"""import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # must be set before CUDA initializes (fresh kernel)

IS_KAGGLE = os.path.exists("/kaggle/input")  # reused in cells 6 and 18 (upload/download) to pick Kaggle vs. Colab I/O

USE_BOTH_GPUS = False  # KEEP False for 4-bit: Trainer refuses to train a 4-bit model split over
    # devices ("can't train a model loaded in 8-bit/4-bit on a different device"). True: split the model's layers across both GPUs (model parallel) -- more total
    # VRAM for activations, so longer MAX_LENGTH fits. NOT faster: one GPU works at a time.
    # False: hide GPU 1 and train on GPU 0 only -- the older, known-working path if the split errors.
    # Either way this must be decided in a FRESH kernel (restart the session before flipping it).

if not USE_BOTH_GPUS:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastVisionModel
import torch

n_gpus = torch.cuda.device_count()
if USE_BOTH_GPUS:
    assert n_gpus >= 2, f"USE_BOTH_GPUS=True but torch sees {n_gpus} GPU(s) -- enable the 2nd GPU in the Kaggle accelerator settings (GPU T4 x2)"
else:
    assert n_gpus == 1, (
        f"torch still sees {n_gpus} GPUs -- CUDA_VISIBLE_DEVICES didn't take effect because CUDA was "
        "already initialized in this kernel. Restart the session and run from this cell again."
    )

BASE_MODEL = "unsloth/gemma-4-E4B-it"  # swap here if this repo id has moved
MAX_LENGTH = 4096  # T4 (no bf16/fp16 for gemma4 -> fp32 activations, 262k-vocab logits) OOMs at 8192; raise only on a bigger GPU.
                    # Re-export with --max-context-chars ~10000 so most rows fit under this. Original note: truncated corpus caps each example at ~24k chars (~6k tokens) of text;
                    # this leaves headroom for chat-template overhead and per-tokenizer variance
                    # from the rough 4-chars/token char-count estimate -- see cell 8's printed
                    # real token-count check

MAX_IMAGE_SIDE = 768   # RAM/CPU guard only: shrink huge screenshots right after decoding. Token cost is
                       # controlled by MAX_SOFT_TOKENS below, not by this.
MAX_SOFT_TOKENS = 70   # visual token budget PER IMAGE (Gemma4ImageProcessor default 280; the Gemma
                       # family's allowed budgets are 70/140/280/560/1120). 140 ~ half the cost.

model, processor = FastVisionModel.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    max_seq_length=MAX_LENGTH,  # Unsloth's default is 2048, and the vision collator truncates to it
    use_gradient_checkpointing="unsloth",
    device_map="balanced" if USE_BOTH_GPUS else {"": 0},  # "balanced": spread layers evenly over all visible GPUs
)
if USE_BOTH_GPUS:
    print("device map:", sorted(set(getattr(model, "hf_device_map", {}).values()), key=str))

processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
processor.image_processor.image_seq_length = MAX_SOFT_TOKENS  # keep in step -- both are 280 by default
print(processor.image_processor)
print(f"GPU memory after load: {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated (MAX_LENGTH={MAX_LENGTH}, soft tokens={MAX_SOFT_TOKENS})")"""),
    md(
"""## 3. Upload the exported corpus

Both from `orchestrator/data/finetune/` (generated by `python -m
orchestrator finetune export-tool-data ... --val-frac 0.2
--max-context-chars 24000`; gitignored — the underlying sessions can carry
job/company/message content and embedded screenshots).

**On Kaggle**: package `tool_select.jsonl` and `tool_select.val.jsonl` into
a private Kaggle Dataset and attach it to this notebook (Notebook > Add
Input). The cell below finds them anywhere under `/kaggle/input/` and
copies them next to the notebook.

**On Colab**: the cell below opens an upload dialog instead — just pick
both files when prompted."""),
    code(
"""if IS_KAGGLE:
    import glob, shutil

    train_matches = glob.glob("/kaggle/input/**/tool_select.jsonl", recursive=True)
    val_matches = glob.glob("/kaggle/input/**/tool_select.val.jsonl", recursive=True)
    assert train_matches and val_matches, (
        "tool_select.jsonl / tool_select.val.jsonl not found under /kaggle/input -- add a "
        "Kaggle Dataset containing both files (Notebook > Add Input), built from this repo's "
        "orchestrator/data/finetune/ locally, then re-run this cell."
    )
    shutil.copy(train_matches[0], "tool_select.jsonl")
    shutil.copy(val_matches[0], "tool_select.val.jsonl")
    print("Found and copied:")
    print(" ", train_matches[0])
    print(" ", val_matches[0])
else:
    from google.colab import files

    print("Select tool_select.jsonl and tool_select.val.jsonl (multi-select in the file dialog):")
    uploaded = files.upload()
    assert "tool_select.jsonl" in uploaded, "missing tool_select.jsonl"
    assert "tool_select.val.jsonl" in uploaded, "missing tool_select.val.jsonl\""""),
    md(
"""## 4. Format for training

Converts our `{"type": "image_url", "image_url": {"url": "data:..."}}`
blocks (the OpenAI-style shape `orchestrator/db.py` already produces for
the live screenshot tool) into the shape Unsloth's vision docs require --
`{"type": "image", "image": <PIL.Image>}`, a real decoded image object,
not a data URI string. Every message's `content` is also normalized to a
list of blocks (`[{"type": "text", "text": ...}]` for a plain string) --
`UnslothVisionDataCollator` expects that structure throughout, not a mix
of bare strings and block lists.

**Memory**: the corpus is *not* loaded into RAM. `LazyJsonlDataset` keeps
only a byte offset per row and reads + decodes one example (images
included) each time the trainer asks for it, so peak RAM is one batch of
decoded screenshots rather than the whole dataset. The diagnostics below
also stream row by row.

**No rows are dropped this round** -- see cell 0's markdown for why. An
orchestrator example can carry an image anywhere in a long, variable-length
prefix (wherever the `screenshot` tool was called earlier in that
session), so `n_with_image` below counts a row as image-bearing if *any*
message in it has one, not just a fixed index.

Also prints the *real* tokenized text length distribution (not the
char-count estimate `--max-context-chars` used) so you can see whether
`MAX_LENGTH` (cell 2) covers this export's text -- but note it's text
only: an image-bearing row's actual sequence is longer than this number
once the vision tower's own tokens are added, and this estimate can't see
that cost."""),
    code(
"""import base64, io, json
from PIL import Image
from torch.utils.data import Dataset

def _decode_data_uri(url):
    _, _, b64data = url.partition(",")
    img = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))  # in-place, keeps aspect ratio, never upscales
    return img

def _to_vision_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for b in content:
        if b.get("type") == "image_url":
            blocks.append({"type": "image", "image": _decode_data_uri(b["image_url"]["url"])})
        else:
            blocks.append(b)
    return blocks

def to_vision_messages(row):
    return [{"role": m["role"], "content": _to_vision_blocks(m["content"])} for m in row["messages"]]

def has_image(row):
    return any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m["content"])
        for m in row["messages"]
    )

class LazyJsonlDataset(Dataset):
    \"\"\"Indexes a JSONL file by byte offset and reads + decodes ONE row per __getitem__.

    Nothing but an int offset per row stays in RAM; screenshots are base64-decoded
    into PIL images only for the rows in the batch currently being collated, then
    garbage-collected. (The previous version decoded every image in both splits up
    front, which is what exhausted RAM.)
    \"\"\"

    def __init__(self, path):
        self.path = path
        self.offsets = []
        self.keep = None  # optional subset of row indices, set by filter_to_max_length
        with open(path, "rb") as fh:
            pos = 0
            for line in fh:
                if line.strip():
                    self.offsets.append(pos)
                pos += len(line)

    def __len__(self):
        return len(self.offsets) if self.keep is None else len(self.keep)

    def raw(self, i):
        j = i if self.keep is None else self.keep[i]
        with open(self.path, "rb") as fh:
            fh.seek(self.offsets[j])
            return json.loads(fh.readline().decode("utf-8"))

    def __getitem__(self, i):
        return {"messages": to_vision_messages(self.raw(i))}

    def filter_to_max_length(self, max_length):
        \"\"\"Drop rows whose REAL token length (image tokens included) exceeds max_length.

        Truncating such a row at train time can slice through an expanded run of
        image tokens, which makes the processor raise "Mismatch in `image` token
        count". Dropping is the safe option; streams one row at a time.
        \"\"\"
        keep, dropped, first_err, lens = [], 0, None, []
        for i in range(len(self)):
            try:
                ids = processor.apply_chat_template(
                    to_vision_messages(self.raw(i)), tokenize=True, return_dict=True)["input_ids"]
                n = len(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else len(ids)
                lens.append(n)
                ok = n <= max_length
            except Exception as e:
                ok = False
                if first_err is None:
                    first_err = f"row {i}: {type(e).__name__}: {e}"
            if ok:
                keep.append(i if self.keep is None else self.keep[i])
            else:
                dropped += 1
        self.keep = keep
        if first_err:
            print("  first tokenize error (rows that error are dropped, NOT just over-length):", first_err)
        if lens:
            lens.sort()
            q = lambda f: lens[min(len(lens) - 1, int(len(lens) * f))]
            print(f"  real token length of ALL {len(lens)} rows (images incl.) -- "
                  f"min {lens[0]}  p10 {q(.1)}  p50 {q(.5)}  p90 {q(.9)}  max {lens[-1]}")
        print(f"{self.path}: kept {len(keep)}, dropped {dropped} over MAX_LENGTH={max_length} (incl. image tokens)")

train_dataset = LazyJsonlDataset("tool_select.jsonl")
val_dataset = LazyJsonlDataset("tool_select.val.jsonl")
train_dataset.filter_to_max_length(MAX_LENGTH)
val_dataset.filter_to_max_length(MAX_LENGTH)

# Streaming diagnostics: one row in memory at a time, nothing retained.
n_with_image = 0
token_lens = []
for ds in (train_dataset, val_dataset):
    for i in range(len(ds)):
        r = ds.raw(i)
        if has_image(r):
            n_with_image += 1
        elif ds is train_dataset:
            try:
                token_lens.append(len(processor.tokenizer.apply_chat_template(r["messages"], tokenize=True)))
            except Exception as e:
                print(f"(skipping text-length diagnostic -- {e})")
                token_lens = None
                break
        del r
    if token_lens is None:
        break
print(f"train: {len(train_dataset)}  val: {len(val_dataset)}  "
      f"({n_with_image} total carry an embedded screenshot -- kept this round, see cell 0's markdown)")

if token_lens:
    token_lens.sort()
    p50, p90, mx = token_lens[len(token_lens) // 2], token_lens[int(len(token_lens) * 0.9)], token_lens[-1]
    over_budget = sum(1 for n in token_lens if n > MAX_LENGTH)
    print(f"text-only real token length -- p50 {p50}  p90 {p90}  max {mx}  (MAX_LENGTH={MAX_LENGTH})")
    if over_budget:
        print(f"{over_budget}/{len(token_lens)} text-only example(s) exceed MAX_LENGTH and will be truncated "
              "by the trainer -- consider raising MAX_LENGTH (cell 2) or lowering --max-context-chars and "
              "re-exporting")

train_dataset[0]"""),
    md(
"""## 5. LoRA adapter

Same config as the browsing notebook, same reasoning: `finetune_vision_layers=False`.
131 image-bearing train examples (cell 4) is better coverage than the
browsing notebook's 22, but still thin for safely adapting a whole vision
tower without overfitting on — or degrading general screenshot
understanding for — a few dozen screenshots. Freezing it still lets images
flow through as features into the LoRA-tuned language/attention/mlp
layers, so this pass gets real signal on "here's a screenshot, here's the
right action" without touching how the model sees."""),
    code(
"""model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)"""),
    md(
"""## 6. Train

372 train examples (exact count printed by cell 4), batch 1 × grad-accum 8
= effective batch 8. 3 epochs is a starting point for a corpus this size —
watch `eval_loss`; if it's still dropping at the end, this is worth a
longer run once there's more exported data, and if it's *rising* before
epoch 3, cut epochs back (this little data overfits fast).

`UnslothVisionDataCollator(model, processor)` -- not the plain
`dataset_text_field="text"` pipeline the browsing notebook (still) uses --
handles chat-templating, tokenizing, and image processing together, per
batch, from the raw `messages` column cell 4 built. That's why
`remove_unused_columns=False` and `dataset_kwargs={"skip_prepare_dataset":
True}` are set below: they tell `SFTTrainer` not to run its own
text-formatting/tokenization ahead of time and strip unknown columns --
the collator needs the untouched `messages` field, not a pre-rendered
`"text"` string. See cell 0's markdown for why this is still full-sequence
loss rather than assistant-only, and the fallback if this errors on the
image/text mix specifically."""),
    code(
"""from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor, max_seq_length=MAX_LENGTH),
    args=SFTConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_LENGTH,
        dataloader_num_workers=0,  # lazy dataset decodes per batch in the main process; keep RAM flat
    ),
)

print(f"GPU memory before train: {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated")
trainer_stats = trainer.train()"""),
    md(
"""## 7. Sanity-check before exporting

Run a few held-out val examples through the fine-tuned model and compare
the ` ```action ` block it produces against the real one from the session
transcript — before the (slow) GGUF export below. Unlike the browsing
notebook's short examples, these prompts can be long; `max_new_tokens=256`
is enough for the reasoning line + action fence, not for echoing back the
prompt.

Uses `processor.apply_chat_template` (the full processor, not
`processor.tokenizer`) — a val prompt here can itself contain an image
block anywhere in its prefix (cell 4's `has_image`), and only the full
processor knows how to turn that into real `pixel_values` alongside the
tokenized text; the plain tokenizer only handles the text side."""),
    code(
"""FastVisionModel.for_inference(model)

import random

sample_examples = [val_dataset[i] for i in random.sample(range(len(val_dataset)), min(5, len(val_dataset)))]

for ex in sample_examples:
    prompt_messages = ex["messages"][:-1]  # everything up to, but not including, the decision
    ground_truth = ex["messages"][-1]["content"]
    if isinstance(ground_truth, list):  # normalized by cell 4 into [{"type": "text", "text": ...}]
        ground_truth = ground_truth[0]["text"]
    inputs = processor.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    predicted = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("=" * 80)
    print("GROUND TRUTH:\\n", ground_truth[-500:])
    print("-" * 80)
    print("PREDICTED:\\n", predicted[:500])"""),
    md(
"""## 8. Save adapter + export merged GGUF

Named distinctly from the browsing notebook's `gemma4-e4b-linkedin-lora` /
`gemma4-e4b-linkedin-merged` — this is a separate fine-tune, not a
continuation of it (see cell 0). `q4_k_m` matches the quantization level
typical for local LM Studio use on limited VRAM."""),
    code(
"""ADAPTER_DIR = "gemma4-e4b-linkedin-tools-lora"
GGUF_DIR = "gemma4-e4b-linkedin-tools-merged"

model.save_pretrained(ADAPTER_DIR)
processor.tokenizer.save_pretrained(ADAPTER_DIR)

model.save_pretrained_gguf(
    GGUF_DIR,
    processor.tokenizer,
    quantization_method="q4_k_m",
)"""),
    md(
"""## 9. Download

The `.gguf` is likely a few GB.

**On Kaggle**: there's no in-session download widget — the cell below
copies it into `/kaggle/working/gguf_export/`. Commit the notebook (Save
Version > Save & Run All) and download the file from that version's
**Output** tab once the run finishes.

**On Colab**: the cell below calls `files.download()` directly. If it
stalls or the session disconnects, mount Drive instead (`from
google.colab import drive; drive.mount('/content/drive')`) and `!cp` the
file there, then download from Drive's own UI."""),
    code(
"""import glob

gguf_files = glob.glob(f"{GGUF_DIR}/*.gguf")
print("Found:", gguf_files)

if IS_KAGGLE:
    import shutil

    out_dir = "/kaggle/working/gguf_export"
    os.makedirs(out_dir, exist_ok=True)
    for f in gguf_files:
        dest = os.path.join(out_dir, os.path.basename(f))
        shutil.copy(f, dest)
        print(f"Copied to {dest}")
    print(
        "Commit this notebook (Save Version > Save & Run All) and grab the file(s) from "
        "that version's 'Output' tab once it finishes -- Kaggle has no in-session "
        "browser download like Colab's files.download()."
    )
else:
    from google.colab import files

    for f in gguf_files:
        files.download(f)"""),
    md(
"""## 10. Bring it back into LM Studio

1. Move the downloaded `.gguf` into LM Studio's models folder, in its own
   subfolder — e.g. `<LM Studio models dir>/local/gemma-4-E4B-linkedin-tools/gemma-4-E4B-linkedin-tools.Q4_K_M.gguf`
   (LM Studio expects `<publisher>/<model-name>/<file>.gguf`). **Don't overwrite**
   the browsing notebook's existing `gemma-4-e4b-linkedin` folder — this is a
   separate checkpoint (see cell 0), and `orchestrator/config.json` still
   needs that one unless you're deliberately switching.
2. In LM Studio, reload the models list and load this model on the local
   server. Confirm its model card still shows vision support after the
   merge — it should, since only LoRA weights on the language side changed.
3. To try it with the orchestrator specifically (without changing your
   default `orchestrator/config.json`), pass `--model` on the command line:
   `python -m orchestrator run --model gemma-4-e4b-linkedin-tools "List the available skills."`
   — check LM Studio's `/v1/models` endpoint or UI for the exact served
   name, it won't necessarily match the folder name above.
4. Confirm it responds sanely, and that it actually emits a well-formed
   ` ```action ` fence, before trusting it with a real cycle or pointing
   `orchestrator/config.json`'s `"model"` field at it by default."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
