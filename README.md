# Sona v1 on MuSP-Bench: evaluation

Evaluation materials for the *Sona v1* technical report (Lyrn, October 2026): the question set, the grader, how the
audio-input baselines were queried, per-question results for every system we ran, and a script that recomputes
every number in the report. Sona itself is not included.

```
musp_eval/questions.py       the 238 audio-only questions, the question block, the two halves
musp_eval/grader.py          strict (and lenient) exact-match grader
musp_eval/audio_baseline.py  how the audio-input models were queried on their own
musp_eval/stats.py           Wilson intervals, exact McNemar test
results/systems.csv          systems, model IDs and settings
results/<system>.csv         per question: id, piece, subset, answered_by, extracted answer, correct
score.py                     recomputes the report's tables from results/
```

## Benchmark

MuSP-Bench [1] (Hugging Face dataset `bryel-labs/MuSP-Bench`, CC BY-NC-SA 4.0). We use the 238 questions that can
be answered from audio alone: 106 performance-only questions (P) and 132 score-or-performance questions (S/P:
general 36, tonality/style 48, composer/title 48), over 24 recordings (18 complete piano works and 6 orchestral
excerpts). Questions that need the score (S and S&P) are excluded.

```python
from huggingface_hub import snapshot_download
snapshot_download("bryel-labs/MuSP-Bench", repo_type="dataset", local_dir="MuSP-Bench")

from musp_eval.questions import load_questions
q = load_questions("MuSP-Bench")   # 238 rows with a `subset` column
```

The three orchestral recordings for pieces 19, 20 and 23 are distributed with the BSED dataset; the MuSP-Bench
README explains how to obtain them.

## Grading

`musp_eval/grader.py`. The answer is the last `Final answer:` line of a reply. It is canonicalized (Unicode
accidentals, case, whitespace, punctuation and markdown, `M:SS` to `MM:SS`, "measure" to "bar", list splitting,
unordered lists sorted) and must then equal one of the benchmark's accepted answers exactly. There is no partial
credit and no enharmonic equivalence; timing tolerance exists only where the accepted answers list it. Every number
in the report uses the strict mode; the lenient mode is for diagnostics only.

```python
from musp_eval.grader import grade_row
grade_row(reply_text, q.set_index("id").loc["musp_92"]).correct
```

## Audio-input baselines

Each question is one request: the system prompt below, the recording of its piece as 16 kHz mono 64 kbit/s MP3
sent inline, and the question block from `musp_eval/questions.py`. Code for both providers is in
`musp_eval/audio_baseline.py`.

```
You are an expert musician listening to a recording of a classical music performance. Answer the question about
this recording. Timestamps refer to the recording as given. Think as needed, then end your reply with one line of
the form
Final answer: <answer>
using exactly the requested answer format.
```

## Systems

| System | Model | Input | Settings |
| --- | --- | --- | --- |
| GPT-6 Astra + Sona | `gpt-6-astra` | Sona | reasoning effort high; OpenAI Batch API |
| GPT-6 Sol + Sona | `gpt-6-sol` | Sona | reasoning effort high; OpenAI Batch API; see notes |
| Claude Opus 5.5 + Sona | `claude-opus-5-5` | Sona | adaptive thinking, effort high |
| Gemini 3.8 Flash + Sona | `gemini-3.8-flash` | audio + Sona | default thinking and temperature; max output 32,768 |
| Qwen3.5-Omni-Plus + Sona | `qwen3.5-omni-plus` | audio + Sona | temperature 0; max output 16,384; see notes |
| Qwen3.5-Omni-Plus | `qwen3.5-omni-plus` | audio | temperature 0; max output 8,192 |
| Gemini 3.8 Flash | `gemini-3.8-flash` | audio | default thinking and temperature; max output 32,768 |
| Qwen3-Omni-Flash | `qwen3-omni-flash` | audio | temperature 0 |
| Qwen3-Omni-30B-A3B | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | audio | open weights on vLLM 0.11.2; temperature 0 |

Qwen models were called through Alibaba Cloud Model Studio (DashScope) by alias, Gemini through the Gemini API.

## Results

`python score.py` prints:

```
Accuracy (%), strict grader, 238 audio-only questions

                               system    P  general  tonality/style  composer/title  all    95% CI
                   GPT-6 Astra + Sona 78.3     77.8            91.7            60.4 77.3 71.6-82.2
               Claude Opus 5.5 + Sona 70.8     66.7            81.2            41.7 66.4 60.2-72.1
                     GPT-6 Sol + Sona 67.9     52.8            85.4            50.0 65.5 59.3-71.3
        GPT-6 Sol + Sona, no deferral 67.9     52.8            85.4            37.5 63.0 56.7-68.9
              Gemini 3.8 Flash + Sona 60.4     61.1            81.2            35.4 59.7 53.3-65.7
             Qwen3.5-Omni-Plus + Sona 29.2     22.2            83.3            45.8 42.4 36.3-48.8
                    Qwen3.5-Omni-Plus 12.3      8.3            81.2            50.0 33.2 27.5-39.4
                     Qwen3-Omni-Flash  4.7     13.9            56.2            27.1 21.0 16.3-26.6
                   Qwen3-Omni-30B-A3B  4.7      5.6            52.1            16.7 16.8 12.6-22.1
                     Gemini 3.8 Flash  9.4      5.6            25.0             4.2 10.9  7.6-15.5
     Muse Spark 1.2 (reported in [1])  3.8     11.1            20.8             2.1  8.0
Audio Flamingo Next (reported in [1])  0.9      0.0            22.9             0.0  5.0

Same model, audio input vs audio input + Sona (paired, exact McNemar)

  Gemini 3.8 Flash   all   10.9 ->  59.7   only audio   5, only + Sona 121, p = 6e-30
  Gemini 3.8 Flash   P      9.4 ->  60.4   only audio   2, only + Sona  56, p = 1.2e-14
  Qwen3.5-Omni-Plus  all   33.2 ->  42.4   only audio  15, only + Sona  37, p = 0.0032
  Qwen3.5-Omni-Plus  P     12.3 ->  29.2   only audio   6, only + Sona  24, p = 0.0014

Development half (odd-numbered pieces) vs held-out half (even-numbered pieces), accuracy (%)

                       system  development all (118)  development P (51)  held-out all (120)  held-out P (55)
           GPT-6 Astra + Sona                   76.3                78.4                78.3             78.2
             GPT-6 Sol + Sona                   59.3                62.7                71.7             72.7
GPT-6 Sol + Sona, no deferral                   57.6                62.7                68.3             72.7
       Claude Opus 5.5 + Sona                   62.7                62.7                70.0             78.2
      Gemini 3.8 Flash + Sona                   60.2                62.7                59.2             58.2
     Qwen3.5-Omni-Plus + Sona                   40.7                25.5                44.2             32.7
            Qwen3.5-Omni-Plus                   32.2                 9.8                34.2             14.5
             Gemini 3.8 Flash                   11.0                 9.8                10.8              9.1
             Qwen3-Omni-Flash                   24.6                 3.9                17.5              5.5
           Qwen3-Omni-30B-A3B                   20.3                 7.8                13.3              1.8
```

## Notes

- **GPT-6 Sol + Sona.** Composer/title questions in this row use Qwen3.5-Omni-Plus's audio-only answers (marked
  in the `answered_by` column). The choice was made on the development half, where Qwen3.5-Omni-Plus on audio
  answered more composer/title questions than GPT-6 Sol with Sona; the same comparison kept every question group
  with the named model for GPT-6 Astra and Claude Opus 5.5. With every question answered by GPT-6 Sol, the row
  scores 63.0% (`results/gpt-6-sol_sona_no-deferral.csv`).
- **Qwen3.5-Omni-Plus + Sona.** On piece 5 (7 questions) the input exceeds the model's input limit, so those
  questions use its audio-only answers (4 of 7 correct; marked in `answered_by`). On the 231 questions that fit:
  42.0% with Sona against 32.5% on audio alone. The first 22 answers were generated with a maximum output of
  8,192 tokens; the three that reached it were asked again with 16,384, the setting used for the rest.
- **Development half.** Design choices were made on the odd-numbered pieces (51 P questions and their S/P
  questions) before the full set was scored; the even-numbered pieces were held out. Both halves are reported above.
- **Training-data overlap.** Some Sona components were trained on public piano datasets that include
  MuSP-Bench's piano recordings, so piano results should be read as an upper bound.
- **Reported rows.** Muse Spark 1.2 and Audio Flamingo Next are MuSP-Bench Table 1 values (counts rounded back from
  the published percentages), produced by the benchmark authors with their prompts and grader. For comparison, our
  Qwen3.5-Omni-Plus audio-only run scores 33.2% (P 12.3%); the paper reports 35.7% (P 14.2%).
- **Runs.** One run per system, September and October 2026, so differences under about 5 points should not be
  read as a ranking.

## Reference

[1] Liessens Dujardin, Yu, and Miao. MuSP-Bench: Advanced Multimodal Benchmarking of Music Understanding Across
Score and Performance. [arXiv:2608.28212](https://arxiv.org/abs/2608.28212), 2026.

## License

Code is released under the MIT License. `results/` holds only MuSP-Bench question IDs and our systems' answers;
the questions and accepted answers belong to their authors (CC BY-NC-SA 4.0) and are not redistributed here.
