# Council calibration: pairs_v2_c0_sonnet_medium_vs_opus

**sonnet/medium rejects 1.0% of 300 pairs items vs opus/xhigh's 26.3% (leniency delta +25.3 pp); 25.6% [21.0%, 30.8%] of sonnet/medium's accepts are rejected by opus/xhigh.**

Reproduce:

```
python3 experiments/d5_cohere_track/training_data/council/compare_councils.py --council pairs --chunk-dir data/generated/v2/council/pair_chunks --a-dir data/generated/v2/council/pair_votes --a-label opus/xhigh --b-dir data/generated/v2/council/pair_votes_sonnet --b-label sonnet/medium --name pairs_v2_c0_sonnet_medium_vs_opus --chunks c0 --by why_flagged
```

## Config

| field | value |
|---|---|
| council | pairs (lenses: coherence, identity, plausibility, register, skeptic) |
| final verdict | ACCEPT at >= 4/5 lenses |
| reference A | opus/xhigh -- `data/generated/v2/council/pair_votes` (stamp: yes) |
| candidate B | sonnet/medium -- `data/generated/v2/council/pair_votes_sonnet` (stamp: yes) |
| corpus | `data/generated/v2/council/pair_chunks`: 1 chunks, 300 items (sha256 per chunk in the JSON) |
| seed | none: deterministic, no sampling |

## Final verdicts (>= 4/5)

| | sonnet/medium accept | sonnet/medium reject | total |
|---|---:|---:|---:|
| **opus/xhigh accept** | 221 | 0 | 221 |
| **opus/xhigh reject** | 76 | 3 | 79 |
| total | 297 | 3 | 300 |

| metric | value |
|---|---|
| reject rate opus/xhigh | 26.3% |
| reject rate sonnet/medium | 1.0% |
| leniency delta (A reject - B reject) | +25.3 pp |
| agreement / Cohen's kappa | 74.7% / 0.055 |
| sonnet/medium false-accept vs opus/xhigh (B accepts that A rejects) | 76/297 = 25.6% [21.0%, 30.8%] |
| sonnet/medium miss rate vs opus/xhigh (A rejects that B accepts) | 76/79 = 96.2% [89.4%, 98.7%] |
| Fleiss' kappa across lenses, opus/xhigh / sonnet/medium | 0.626 / -0.006 |

## Per lens

| lens | n | opus/xhigh accept | sonnet/medium accept | leniency (pp) | Cohen's kappa | only opus/xhigh rejects | only sonnet/medium rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherence | 300 | 75.7% | 100.0% | +24.3 | 0.000 | 73 | 0 |
| identity | 300 | 78.0% | 99.0% | +21.0 | 0.069 | 63 | 0 |
| plausibility | 300 | 84.0% | 100.0% | +16.0 | 0.000 | 48 | 0 |
| register | 300 | 85.3% | 100.0% | +14.7 | 0.000 | 44 | 0 |
| skeptic | 300 | 76.0% | 83.7% | +7.7 | 0.436 | 39 | 16 |

## sonnet/medium at other thresholds (vs opus/xhigh at >= 4)

Calibration data only. A threshold picked from this table must be validated on a separate audit sample of the bulk run (`sample_audit.py`), not on these chunks.

| sonnet/medium min agree | sonnet/medium accept | agreement | kappa | sonnet/medium false-accept vs opus/xhigh | sonnet/medium miss rate |
|---:|---:|---:|---:|---|---|
| 1 | 100.0% | 73.7% | 0.000 | 79/300 = 26.3% [21.7%, 31.6%] | 79/79 = 100.0% [95.4%, 100.0%] |
| 2 | 100.0% | 73.7% | 0.000 | 79/300 = 26.3% [21.7%, 31.6%] | 79/79 = 100.0% [95.4%, 100.0%] |
| 3 | 100.0% | 73.7% | 0.000 | 79/300 = 26.3% [21.7%, 31.6%] | 79/79 = 100.0% [95.4%, 100.0%] |
| 4 | 99.0% | 74.7% | 0.055 | 76/297 = 25.6% [21.0%, 30.8%] | 76/79 = 96.2% [89.4%, 98.7%] |
| 5 | 83.7% | 78.7% | 0.374 | 47/251 = 18.7% [14.4%, 24.0%] | 47/79 = 59.5% [48.5%, 69.6%] |

## By `why_flagged`

| why_flagged | n | opus/xhigh reject | sonnet/medium reject | leniency (pp) | sonnet/medium miss rate |
|---|---:|---:|---:|---:|---|
| vowel dropped beside a vowel: judge per pair | 55 | 76.4% | 3.6% | +72.7 | 40/42 = 95.2% [84.2%, 98.7%] |
| word-initial ع-deletion: judge per pair | 245 | 15.1% | 0.4% | +14.7 | 36/37 = 97.3% [86.2%, 99.5%] |

## Final-verdict disagreements (76; first 40 shown, all in the JSON)

| sid | text | opus/xhigh | sonnet/medium | opus/xhigh rejected on | sonnet/medium rejected on |
|---|---|---|---|---|---|
| 2a2dea->2a2da | 2a2dea -> 2a2da | reject 1/5 | accept 5/5 | coherence: Loses the -iyya ending; 2a2da reads like qada (fate/judiciary), not … | - |
| 2a2dea->2a2dah | 2a2dea -> 2a2dah | reject 1/5 | accept 5/5 | coherence: Loses the -iyya ending; 2a2dah no longer reads as qadiyya (issue).; … | - |
| 2a2dea->a2da | 2a2dea -> a2da | reject 1/5 | accept 5/5 | coherence: Loses the -iyya ending; a2da reads like a2di/ada, not qadiyya (issue… | - |
| 2ablia->2abla | 2ablia -> 2abla | reject 0/5 | accept 5/5 | coherence: 2abla reads as 'before her' or 'kiss', not qabiliyya (susceptibility… | - |
| 2ablia->2ablah | 2ablia -> 2ablah | reject 0/5 | accept 5/5 | coherence: 2ablah reads as 'before him', not qabiliyya (susceptibility).; ident… | - |
| 2adea->2ada | 2adea -> 2ada | reject 0/5 | accept 4/5 | coherence: 2ada reads as qada2 (judiciary), not qadiyya (lawsuit).; identity: 2… | skeptic: collides with 2ada/qada (spent/fate), a different real word |
| 2adea->2adaa | 2adea -> 2adaa | reject 0/5 | accept 5/5 | coherence: 2adaa reads as qada2 (judiciary), not qadiyya (case).; identity: 2ad… | - |
| 2adea->2adah | 2adea -> 2adah | reject 0/5 | accept 5/5 | coherence: Loses the -iyya ending; 2adah no longer reads as qadiyya (lawsuit).;… | - |
| 2adey->2adi | 2adey -> 2adi | reject 3/5 | accept 5/5 | identity: 2adi is the usual spelling of قاضي (judge), not أدّى.; skeptic: 2adi … | - |
| 2aema->2ama | 2aema -> 2ama | reject 0/5 | accept 4/5 | coherence: 2ama reads as qama (stature), not qa2ima/qima (list/value).; identit… | skeptic: close to 2ama/qama (stature, height), a different word |
| 2aema->2amaa | 2aema -> 2amaa | reject 0/5 | accept 4/5 | coherence: 2amaa reads as qama (stature), not qima (value).; identity: 2amaa re… | skeptic: same, close to 2ama/qama (height) |
| 2aema->ama | 2aema -> ama | reject 0/5 | accept 4/5 | coherence: ama reads as amma/omma; 'mn ama el-eeman' loses 'value of faith'.; i… | skeptic: collides with ama/amma (as-for/but), a common discourse word |
| 2aids->2ads | 2aids -> 2ads | reject 0/5 | accept 5/5 | coherence: Diphthong lost; el-2ads reads like el-2ods (Jerusalem), not AIDS.; i… | - |
| 2aima->2ama | 2aima -> 2ama | reject 0/5 | accept 4/5 | coherence: 2ama reads as qama (stature), not qima (value).; identity: 2ama read… | skeptic: close to 2ama/qama (height), a different word |
| 2aima->ama | 2aima -> ama | reject 0/5 | accept 4/5 | coherence: ama reads as amma (as for), not qima (value).; identity: ama reads a… | skeptic: collides with ama/amma (as-for/but), a common word |
| 2albia->2alba | 2albia -> 2alba | reject 0/5 | accept 5/5 | coherence: 2alba reads as 'her heart', not 2albiyya (cardiac); 'heart diseases'… | - |
| 2albia->2albaa | 2albia -> 2albaa | reject 0/5 | accept 5/5 | coherence: 2albaa reads as 'her heart', not the adjective 2albiyya (cardiac).; … | - |
| 2albia->alba | 2albia -> alba | reject 0/5 | accept 4/5 | coherence: alba loses the nisba ending; not recoverable as 2albiyya (cardiac).;… | skeptic: Alba is a well-known Egyptian appliance brand name |
| 2almania->2almana | 2almania -> 2almana | reject 2/5 | accept 5/5 | coherence: Loses -ia; 2almana reads like el-amana (trust), not Germany/German.;… | - |
| 2almania->almana | 2almania -> almana | reject 1/5 | accept 5/5 | coherence: almana reads as el-amana (trust/secretariat), not Germany/German.; p… | - |
| 2asasia->2asasa | 2asasia -> 2asasa | reject 0/5 | accept 4/5 | coherence: Loses the nisba -iyya; 2asasa reads like asaso (its foundation), not… | skeptic: close to asasan (basically, adverb), a different word/part of speech |
| 2asasia->asasa | 2asasia -> asasa | reject 0/5 | accept 4/5 | coherence: Loses the nisba -iyya; asasa no longer means 'basic/essential'.; ide… | skeptic: same, close to asasan (basically) |
| 2ay->2i | 2ay -> 2i | reject 2/5 | accept 5/5 | coherence: In the vomiting example, 2i does not read as 2ay2; ayy is not ee.; p… | - |
| 2ayd->2id | 2ayd -> 2id | reject 2/5 | accept 4/5 | coherence: '3ala 2id el7aya' reads as 3ala eid (at the hand of), not 2ayd.; ide… | skeptic: collides with 2eid/3eid (Eid holiday), a very common word |
| 2ayda->2ida | 2ayda -> 2ida | reject 2/5 | accept 4/5 | coherence: 2ida reads as eidha (her hand), not 2ayd (under research).; identity… | skeptic: same Eid-word collision risk |
| 2eamet->2ameet | 2eamet -> 2ameet | reject 0/5 | accept 5/5 | coherence: Vowel moved; 2ameet no longer reads as 2eemet (value of).; identity:… | - |
| 2eamet->2amet | 2eamet -> 2amet | reject 0/5 | accept 4/5 | coherence: 2amet reads as 2amet (she stood up), not 2eemet (value of).; identit… | skeptic: collides with 2amet/qamet (she stood up), a common verb |
| 2eamet->2amit | 2eamet -> 2amit | reject 0/5 | accept 4/5 | coherence: 2amit reads as 2amet (she stood up), not 2eemet (value of).; identit… | skeptic: same, close to qamet (stood up) |
| 2eamet->amet | 2eamet -> amet | reject 0/5 | accept 4/5 | coherence: amet reads as ommet/2amet, not 2eemet (value of).; identity: amet re… | skeptic: casually pronounced like qamet (stood up) even without hamza |
| 2eima->2ema | 2eima -> 2ema | reject 3/5 | accept 5/5 | identity: 2ema reads as قمة (summit) or إما (either), not قيمة.; skeptic: 2ema … | - |
| 2eima->ima | 2eima -> ima | reject 2/5 | accept 4/5 | coherence: ima collides with imma (either); 'bit2adr ima el2ersh' loses 'value'… | skeptic: collides with ema/imma (either), a common word |
| 2esay->2esi | 2esay -> 2esi | reject 0/5 | accept 5/5 | coherence: 2esi reads like 2asi (harsh); the base noun is lost.; identity: ay->… | - |
| 2etrain->2etran | 2etrain -> 2etran | reject 0/5 | accept 5/5 | coherence: 2etran reads as 2etran (tar), not train.; identity: 2etran reads as … | - |
| 2etrain->etran | 2etrain -> etran | reject 0/5 | accept 5/5 | coherence: etran reads as 2atran (tar), not train.; identity: etran loses the a… | - |
| 2ey->2i | 2ey -> 2i | reject 3/5 | accept 5/5 | plausibility: ey (pronounced ay) written as bare i is not a real spelling of ay… | - |
| 2ezay->2ezi | 2ezay -> 2ezi | reject 2/5 | accept 5/5 | coherence: 2ezi reads as English 'easy' ('masha easy'); loses stressed -ay of e… | - |
| 2onlain->2onlaan | 2onlain -> 2onlaan | reject 2/5 | accept 5/5 | identity: 2onlaan loses the ai of 'online'; not recognisable as online.; plausi… | - |
| 2onlain->2unlan | 2onlain -> 2unlan | reject 1/5 | accept 4/5 | coherence: Two changes with bare context; 2unlan no longer recognizable as 'onl… | skeptic: close to 2e3lan (announcement/ad), a different common word |
| 32l->a2l | 32l -> a2l | reject 2/5 | accept 4/5 | coherence: a2l collides with a2al (less): 'ma3 a2l mt2bl' misreads.; identity: … | skeptic: collides with a2l/a2al (less), a very common word |
| 32ol->a2ol | 32ol -> a2ol | reject 1/5 | accept 4/5 | coherence: a2ol collides with a2ool (I say), not 3o2ool (minds).; identity: a2o… | skeptic: collides with a2ol/a2oul (I say), a common verb |
