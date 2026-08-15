# Benchmark result status

Historical and Current are generation-provenance classifications. The exact
filename and SHA-256 digest pin the bytes to which a classification applies;
they do not establish that classification by themselves. The Historical rows
below record results known to come from an invalidated generation path. These
files are not authoritative outputs of the current scripts and must not
support a new quantitative claim until regenerated.

| Artefact | SHA-256 | Status | Reason |
|---|---|---|---|
| `results_rerank_dev.json` | `25ec3ac179a035526c89e577d252e86087b90e6438970668762457e5643f4340` | Historical | Re-ranking used only the first 500 generated candidates. |
| `results_rerank_test.json` | `126ba2ff15eaf68f44490600c00445ae1119d14e096a77ef3736707f9eaa11a6` | Historical | Re-ranking used only the first 500 generated candidates. |
| `results_stats_test_k1.json` | `9c592596940841ee2309024b262971f1b16edbcd87eda83a05fe04b2ac59cc18` | Historical | Statistics replayed the capped candidate dump. |
| `results_stats_test_k5.json` | `15976a34e3e6c6649d84c01603b1177843bceb07ec2138221c8fff95965b1c00` | Historical | Statistics replayed the capped candidate dump. |
| `results_stats_test_k20.json` | `db4643d7bc4705e07991092ad4f4c523297ceb7bdc83892c18e579364411620e` | Historical | Statistics replayed the capped candidate dump. |
| `results_lm_dev.json` | `9952f8fa25e5d11345d875a0e42048a6db0497fe6ad62cec563f792744cc9601` | Historical | The dump was capped, and unresolved context tokens were removed rather than represented in position. |
| `results_lm_dev_ddb.json` | `0a195242b7dff755fead7e3398c6820d53e2f803ea1df5cc743c935ddd4fe28c` | Historical | The dump was capped; unresolved context and DDbDP training tokens were removed rather than represented in position. |
| `results_lm_dev_glaux30.json` | `9ac0a7b30201709f8ae6dc392ea9c14d1d68c79e38b913e407ed2adfd8c35981` | Historical | The dump was capped, and unresolved context tokens were removed rather than represented in position. |
| `results_lm_dev_glauxfull.json` | `c5f244b02c3de6d2992aab48ed517aa0c8afa909a13af1a81d74630973887692` | Historical | The dump was capped, and unresolved context tokens were removed rather than represented in position. |
| `results_lm_dev_glauxfull_plus_ddb.json` | `6a7c7a8ea1e779a8ec11ea26cbff30e8f954421992eb7332c89392395e4b92a3` | Historical | The dump was capped; unresolved context and DDbDP training tokens were removed rather than represented in position. |
| `results_lm_dev_glauxmatched.json` | `ddb2d46282a17be3442b889f9eade5fbee20a88ea22d0933a40b46310d48a2c8` | Historical | The dump was capped, and unresolved context tokens were removed rather than represented in position. |
| `results_lm_paired.json` | `39cf00907f0bbdab31a76b10bb184867ff4317da96653aa82a156e58ccd2e344` | Historical | The dump was capped; unresolved context and DDbDP training tokens were removed rather than represented in position. |
| `results_lm_test.json` | `affb4fe015c9348e3c84cd1af55312391a10f68fbb334c9517f29c8415679dad` | Historical | The dump was capped, and unresolved context tokens were removed rather than represented in position. |
| `results_facet_dev.json` | `a462f2c52a8b9c9c77e8137b0d8515f5effd266151ef95aaf6d2bd65e88af0d2` | Historical | The analysis replayed the capped candidate dump. |
| `results_facet_test.json` | `dd96cced405458d14ddec53cea68e78b5356fc68b90d198d77fc1f5493ec650d` | Historical | The analysis replayed the capped candidate dump. |
| `results_altprob_dev.json` | `5dbde98cc6b1e0b68a7ee713b66fa00b4c3ac2a8907dfb5701629f3112c1b260` | Historical | The analysis replayed the capped candidate dump. |
| `results_altprob_test.json` | `9913e4f785e461c537d3989235b102a7227512432022d5a141f2987a916bbd96` | Historical | The analysis replayed the capped candidate dump. |
| `results_zones_dev.json` | `abdb08f6efafb14a5e03b2f18bb74095792fbd5ad8c65088dee71a6b15ca977f` | Historical | The analysis replayed the capped candidate dump. |
| `results_zones_dev_k20.json` | `46397050099d4b42467969d3d3b1f4677939d85d4f46575f0ca21cf185e04d1c` | Historical | The analysis replayed the capped candidate dump. |
| `results_zones_dev_k50.json` | `a8eebb4931dfe4934fef937e63756f0caa4ae6be5a6695e6803ae4a6f898e890` | Historical | The analysis replayed the capped candidate dump. |
| `results_propnoun_test.json` | `0644683257e6312f1a69ed0d8e8534578d28d62457d6f30137eda2d321d783a5` | Historical | Its bucket calculations depend on the capped dump. |
| `results_decision_B3u.json` | `65bd1f75ce160584f2dfb3c3c739822d6212b427b21de6770ac9e06b64a0fdc7` | Historical | Its bucket calculations depend on the capped dump. |
| `results_coverage_fit.json` | `8d246ce5daec9f2d280b797ce67f521ef7a5cce882c07a75c05023ffef34bd7b` | Historical | Its coverage calculations depend on the capped dump or old context representation. |

## Current

There are no Current artefacts. A result enters this section only after its
embedded provenance identifies the benchmark code and every material input.

## Unclassified

| Artefact | SHA-256 | Status | Reason |
|---|---|---|---|
| `results_b0.json` | `cc8c6abb91424faab5aaef705ce4a9cb3180969aace92ac300a064824a3bf3e1` | Unclassified | Pre-provenance artefact: it was computed in memory, but the generating code and input identities were not recorded. |
| `results_ddb_dev_B3u.json` | `7dda26e964356732e4f580bf43040545743dcb6da439e669bd5e6cd19777c751` | Unclassified | Pre-provenance artefact: it was computed in memory, but the generating code and input identities were not recorded. |
| `results_bert_dev.json` | `fcb4a08f19d405663b0871370b33c7ccfa63c4fe25f7b2df276a9338a3947b23` | Unclassified | Replays `dump_b3u_feat.jsonl`, whose cap status is not established here. |
| `results_bert_C_dev.json` | `e8ef42e2d197b07e5b17805c02964aefed3231297d4230933bd0b3c7979c17c1` | Unclassified | Same. |
| `results_bert_gate.json` | `a574609726ddb2b9fda6b939740414ccfbd1e8d5c224a855a7e597254f2d9e72` | Unclassified | Same. |

Unclassified is not a weaker Current. These files replay a dump the way the
Historical ones do, and until whoever generated `dump_b3u_feat.jsonl` records
whether it carried the cap, they support no quantitative claim at all.

## Rules

A file is Historical only when its filename and digest select a Historical row
and that row's recorded generation provenance identifies the invalidated
generation path. A regenerated file is not Historical merely because its name
or digest still matches. It remains uncitable until its generation metadata is
validated and this manifest records its status. This applies even when
regeneration reproduces the same bytes: update the entry to Current with the
new `generation_commit` (or equivalent embedded provenance) in the same
change. Current results require the same provenance evidence; filename and
digest alone are insufficient. Ladder results computed directly from the
complete in-memory candidate list are unaffected by the dump cap, but the
dataset generation noted in the accompanying report must still be verified
before citation.

To establish a new result of record:

1. Generate a fresh B3u dump with the current `rlb_ladder.py`; every row must
   satisfy `n_cand == len(cands)`.
2. Regenerate `context_lemmas.json` with the current `rlb_lm.py context`; an
   unresolved token is represented as `<unk>` when contexts are assembled.
3. If a DDbDP LM condition is involved, regenerate `ddb_lemmas/` so every
   unresolved training token is represented by `<unk>` rather than deleted.
4. Re-run the selected dev analysis, freeze its configuration, then score test
   once and regenerate every downstream statistic quoted in a report.
5. Update this document immediately after regeneration: replace the affected
   manifest entries with their new status, SHA-256 digest, generation commit
   SHA (or equivalent embedded generation metadata), and validity rationale,
   even when regeneration produces the same digest.

The large GLAUx and DDbDP lemma corpora are not present in this checkout, so the
historical numbers are deliberately not rewritten or approximated here.
