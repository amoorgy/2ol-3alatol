# Council calibration: fused_v2_glm53_vs_opus_xhigh

**glm-5.3 rejects 20.1% of 1,122 fused items vs opus/xhigh's 37.0% (leniency delta +16.8 pp); 27.8% [25.0%, 30.8%] of glm-5.3's accepts are rejected by opus/xhigh.**

> **WARNING.** panel B label 'glm-5.3' disagrees with its stamp glm-5.3/default

Reproduce:

```
python3 experiments/d5_cohere_track/training_data/council/compare_councils.py --council fused --chunk-dir data/generated/v2/council/fused_chunks --a-dir data/generated/v2/council/fused_votes --a-label opus/xhigh --b-dir data/generated/v2/council/fused_votes_glm --b-label glm-5.3 --name fused_v2_glm53_vs_opus_xhigh --require-complete
```

## Config

| field | value |
|---|---|
| council | fused (lenses: anchor, coherence, degeneracy, fluency, register) |
| final verdict | ACCEPT at >= 4/5 lenses |
| reference A | opus/xhigh -- `data/generated/v2/council/fused_votes` (stamp: none) |
| candidate B | glm-5.3 -- `data/generated/v2/council/fused_votes_glm` (stamp: yes) |
| corpus | `data/generated/v2/council/fused_chunks`: 10 chunks, 1,122 items (sha256 per chunk in the JSON) |
| seed | none: deterministic, no sampling |

## Final verdicts (>= 4/5)

| | glm-5.3 accept | glm-5.3 reject | total |
|---|---:|---:|---:|
| **opus/xhigh accept** | 647 | 60 | 707 |
| **opus/xhigh reject** | 249 | 166 | 415 |
| total | 896 | 226 | 1,122 |

| metric | value |
|---|---|
| reject rate opus/xhigh | 37.0% |
| reject rate glm-5.3 | 20.1% |
| leniency delta (A reject - B reject) | +16.8 pp |
| agreement / Cohen's kappa | 72.5% / 0.348 |
| glm-5.3 false-accept vs opus/xhigh (B accepts that A rejects) | 249/896 = 27.8% [25.0%, 30.8%] |
| glm-5.3 miss rate vs opus/xhigh (A rejects that B accepts) | 249/415 = 60.0% [55.2%, 64.6%] |
| Fleiss' kappa across lenses, opus/xhigh / glm-5.3 | 0.121 / 0.092 |

## Per lens

| lens | n | opus/xhigh accept | glm-5.3 accept | leniency (pp) | Cohen's kappa | only opus/xhigh rejects | only glm-5.3 rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor | 1,122 | 84.0% | 89.3% | +5.3 | 0.480 | 98 | 38 |
| coherence | 1,122 | 81.2% | 89.0% | +7.8 | 0.388 | 132 | 44 |
| degeneracy | 1,122 | 85.2% | 94.2% | +9.0 | 0.221 | 133 | 32 |
| fluency | 1,122 | 32.4% | 55.5% | +23.1 | 0.367 | 314 | 55 |
| register | 1,122 | 87.2% | 93.5% | +6.2 | 0.088 | 125 | 55 |

## glm-5.3 at other thresholds (vs opus/xhigh at >= 4)

Calibration data only. A threshold picked from this table must be validated on a separate audit sample of the bulk run (`sample_audit.py`), not on these chunks.

| glm-5.3 min agree | glm-5.3 accept | agreement | kappa | glm-5.3 false-accept vs opus/xhigh | glm-5.3 miss rate |
|---:|---:|---:|---:|---|---|
| 1 | 99.9% | 63.1% | 0.003 | 414/1121 = 36.9% [34.2%, 39.8%] | 414/415 = 99.8% [98.7%, 100.0%] |
| 2 | 99.2% | 63.8% | 0.027 | 406/1113 = 36.5% [33.7%, 39.4%] | 406/415 = 97.8% [95.9%, 98.9%] |
| 3 | 93.0% | 68.7% | 0.194 | 344/1044 = 33.0% [30.2%, 35.9%] | 344/415 = 82.9% [79.0%, 86.2%] |
| 4 | 79.9% | 72.5% | 0.348 | 249/896 = 27.8% [25.0%, 30.8%] | 249/415 = 60.0% [55.2%, 64.6%] |
| 5 | 49.5% | 69.8% | 0.397 | 94/556 = 16.9% [14.0%, 20.2%] | 94/415 = 22.7% [18.9%, 26.9%] |

## Final-verdict disagreements (309; first 40 shown, all in the JSON)

| sid | text | opus/xhigh | glm-5.3 | opus/xhigh rejected on | glm-5.3 rejected on |
|---|---|---|---|---|---|
| fs_2a2ema_02 | embare7 katabt 2a2ema gdeeda lel 7agat elli lazem ashtraha men el so2… | reject 3/5 | accept 4/5 | degeneracy: Garbled trailing fragment 'akammel shy el yom' tacked onto the end.… | fluency: 'akammel shy el yom' is garbled; 'men el so2' odd tail; sentence doesn… |
| fs_2a2ema_05 | yalla ne3mel 2a2ema lel agaza ma3 el 3ela 3ashan nertab ezay han3adi … | accept 5/5 | reject 3/5 | - | degeneracy: Trailing fragment 'ezay han3adi el sabt wel 2ad' — '2ad' looks clip… |
| fs_2a2em_09 | ya 3am shoghlok el gedid 2a2em 3ala el safar kteer, 3ashan keda magab… | reject 3/5 | accept 4/5 | fluency: 'magabelnash' is not a real form; '2a2em 3ala el safar' reads MSA.; re… | fluency: 'shoghlok 2a2em 3ala el safar kteer' is an MSA calque; not an Egyptian… |
| fs_2a2em_05 | el khotta 2a2em 3ala ra2y ommy, yalla nettaf2 ma3 el 3a2ila 3ala agaz… | reject 3/5 | accept 4/5 | fluency: Gender error: 'el khotta 2a2em' should be feminine (2ayma).; register:… | fluency: 'el khotta 2a2em 3ala ra2y ommy' is MSA calque; unnatural in Egyptian … |
| fs_2adra_13 | Ana asfa gedan eny neseet el maw3id, ma kontesh 2adra aftakaro, mota3… | reject 2/5 | accept 5/5 | degeneracy: Garbled non-word 'mota3mil' where 'momken' was intended.; fluency: … | - |
| fs_dayra_03 | Momken tesa3edni fi wara2 el dayra 3ashan ana mesh 3aref a5alleso wah… | reject 3/5 | accept 4/5 | fluency: 'wara2 el dayra' is not an Egyptian collocation for department paperwo… | coherence: 'wara2 el dayra' unclear; 'dayra' papers nonsense — circle/departmen… |
| fs_2al_07 | Samahny et2akhar, el zar3a kanet kahla gedan we el mudeer 2al estanna… | reject 1/5 | accept 4/5 | coherence: Excuse for lateness 'el zar3a kanet kahla gedan' is nonsensical; no … | fluency: 'el zar3a kanet kahla' is unclear/unidiomatic; reads as filler nonsens… |
| fs_2al_01 | bokra lazem nrouh el souq el sobh 3ashan ahmed 2al en el mawqe3 hayft… | reject 3/5 | accept 5/5 | fluency: 'el mawqe3 hayfta7' calque/odd for the market opening.; register: 'el … | - |
| fs_2awla_09 | zamaan ma shofnash, ta3ala 2awla en ne7ki w enta teshra7li kull ili 7… | reject 2/5 | accept 4/5 | anchor: 'ta3ala 2awla en ne7ki' is strained; reads like awwalan 'first', not 'm… | fluency: 'ta3ala 2awla en ne7ki' is a calque of أولاً أن نتكلم; not Egyptian ph… |
| fs_3abr_09 | ana b2ali zamaan ma shuftak, bas 3abr el-telefon 3reft akhbarak, ezay… | reject 2/5 | accept 4/5 | degeneracy: Repeats 'akhbarak' twice; formulaic greeting tacked onto end.; flue… | degeneracy: 'akhbarak' repeated twice; two greeting fragments glued together. |
| fs_3abr_08 | akolt koshary 3abr el shari3 embareh ma3 sa7by w el akl kan tayyeb awi | reject 2/5 | accept 4/5 | anchor: English calque 'across the street' as a location; Arabic 3abr doesn't m… | fluency: 'akolt koshary 3abr el shari3' wrong preposition collocation; Egyptian… |
| fs_3abr_12 | kol el ayyam bas7a badri w baghsel weshi w babos el shams 3abr el-sho… | accept 4/5 | reject 3/5 | fluency: '3abr el-shobak' formal calque; Egyptian says 'men el shebbak'. | degeneracy: Template morning-routine list shape, duplicated across items 7, 19,… |
| fs_3ebra_06 | el khabar da asa2ny awi, bas akhodt menno 3ebra 7elwa fel 7ayah w ba2… | reject 2/5 | accept 5/5 | degeneracy: Garbled 'asa2ny' and incoherent trailing 'ba2a 3aref ra7ty'.; fluen… | - |
| fs_3ebra_12 | kol youm basa7a 6:30 badry we da 3ebra en el nom el badry beyfara2 aw… | accept 4/5 | reject 3/5 | fluency: 'da 3ebra en el nom...' calques 'this is a lesson that'. | degeneracy: Template morning-routine sentence, near-duplicate of items 7, 16, 2… |
| fs_3adam_08 | el akl kan tayeb awi bas kan feh 3adam el mal7 fa ma3agabetneesh. | reject 2/5 | accept 4/5 | coherence: Says the food was very good, then that they didn't like it; self-con… | register: '3adam el mal7' is fusha construct; Egyptians would say 'mal7 msh maw… |
| fs_3adam_12 | Ba2oom badri raghm 3adam elnoom, w ashrab 2ahwa 2abl ma akammel mawdo… | accept 4/5 | reject 3/5 | fluency: 'raghm 3adam elnoom' is MSA phrasing, not chat Egyptian. | degeneracy: Template morning-routine fragment, duplicated pattern across other … |
| fs_3alya_03 | ya sa7by momken tes3edny w eshterly el karton ely 3alya el raf 3ashan… | reject 2/5 | accept 4/5 | anchor: 'el karton ely 3alya el raf' reads as garbled 3ala 'on'; neither high n… | anchor: 3alya = preposition '3ala' (on the shelf), not the adjective عالية 'hig… |
| fs_3alya_11 | el match el bare7 kan gamed awi, w nasbet el hodour fel stadium 3alya… | reject 2/5 | accept 5/5 | degeneracy: English 'stadium' unnatural here; an Egyptian would say 'el estad'.… | - |
| fs_3alya_02 | embare7 lama shreit el kotob, la2et el se3r 3alya awi we mal2etsh flu… | reject 2/5 | accept 5/5 | coherence: Says they bought the books yet lacked enough money to pay; contradic… | - |
| fs_leehom_02 | embare7 shoftohom beyet7asabo 3ala el floos, w kano leehom 7afla bel … | accept 4/5 | reject 3/5 | fluency: 'kano leehom 7afla' unnatural; Egyptian: 'kan 3andohom 7afla'. | coherence: Fighting over money then 'they had a party' is a non sequitur with n… |
| fs_leehom_06 | ya3ni el khabar dah kollo 3ashan leehom wala eh, mawdo3 msh ma32ool! | reject 2/5 | accept 5/5 | anchor: '3ashan leehom wala eh' is garbled; sense of leehom unclear.; coherence… | - |
| fs_leehom_15 | ana 2oltelhom 3an el mochkla fel shoghl bas leehom ma2olsh 7aga w m7a… | reject 1/5 | accept 4/5 | anchor: 'bas leehom ma2olsh 7aga' misuses leehom as subject 'they'; neither 'to… | anchor: leehom here is subject pronoun 'but they', not dative 'to them/they hav… |
| fs_3alena_10 | Estanna hena shwaya, lamma yeb2a el dor 3alena ned5ol el 3arabeya be … | accept 4/5 | reject 2/5 | fluency: 'ned5ol el 3arabeya be rasa' is unclear, not natural Egyptian. | degeneracy: Trailing 'be rasa' is garbled, incoherent fragment not natural Egyp… |
| fs_lena_09 | fatra taweela ma shoftak, lena 3andena keteer kalemat w akhbar 3ayzee… | reject 1/5 | accept 4/5 | anchor: 'lena 3andena' doubles possession; lena's role is garbled and strained.… | fluency: lena 3andena حشو وكalemat جمع فصيح؛ الجملة متعثرة |
| fs_3om2_06 | el khabar dah 3ando 3om2 kibeer gedan, we mabsut enni sm3to lessa abl… | accept 4/5 | reject 3/5 | fluency: 'el khabar 3ando 3om2' is an English calque. | anchor: 'Khabar 3ando 3om2' is strained; Egyptians wouldn't naturally use 3om2 … |
| fs_3om2_10 | emshi 3ala el share3 da l7ad el mabnii elly maytoh 3om2 mitr ta7t el … | reject 2/5 | accept 5/5 | coherence: Landmark 'building whose water is a meter deep underground' is nonse… | - |
| fs_3amal_06 | wallahi el khabar da gamed bas el 3amal da mesh sahl khales we lazem … | reject 2/5 | accept 4/5 | degeneracy: Garbled non-word 'nolkaf'.; fluency: 'nolkaf' not a word; '7ata' MS… | fluency: 'nolkaf 7ata nehyto' is garbled/unnatural; sentence doesn't parse as E… |
| fs_3amal_14 | el se3r beta3 el 3amal elly hena kam, 7ad y2olli 3al 7aga di? | reject 3/5 | accept 4/5 | anchor: Shopping 'se3r el 3amal elly hena' -- unclear what 3amal denotes; strai… | fluency: 'el se3r beta3 el 3amal' is an odd collocation; unclear what work pric… |
| fs_3amal_08 | embare7 akalna akla laziza ma3 as7abna, w kanet 3amal 7elw men mama | accept 4/5 | reject 3/5 | fluency: 'kanet 3amal 7elw men mama' calque with gender mismatch. | anchor: '3amal 7elw men mama' means a dish she made, a verb/nominal of making, … |
| fs_ana_12 | kol youm ana ba2oom badri w ashrab 2ahwa w bastahamma w 2afta7 3ala t… | reject 3/5 | accept 4/5 | degeneracy: List-shaped w-chain; garbled '2afta7 3ala tol 3ala el ftoor'.; flue… | fluency: '2afta7 3ala tol 3ala el ftoor' is a calque of 'tuck into breakfast', … |
| fs_ana_10 | khod el share3 da lama tekallem we ana hawarrik el makan 3ala el yami… | reject 2/5 | accept 4/5 | coherence: 'Take this street when you talk' is nonsensical as a direction.; deg… | fluency: 'khod el share3 lama tekallem' is a calque of 'take the street when I … |
| fs_3andha_00 | ana lissa wasalt el beit ba3d yom ta3ban awy w sa7bety 3andha moshkel… | reject 3/5 | accept 5/5 | degeneracy: Three unrelated clauses glued together with 'w'.; fluency: 'yom ta3… | - |
| fs_3andoh_00 | ana lissa wasalt el beit ba3d yom taweel we 3ayyan awi, 3andoh el wa2… | reject 2/5 | accept 4/5 | coherence: Unintroduced 'he has time to listen and yirawwini' doesn't connect t… | fluency: Pronoun shift to '3andoh' incoherent; no clear referent for 'he' in co… |
| fs_3anha_02 | Sa2alni 3anha embare7 fel share3 bas ma 3reftesh ashra7lol 3ashan kan… | reject 3/5 | accept 5/5 | degeneracy: Garbled tokens 'ashra7lol' and 'mad3oo3'.; fluency: 'ashra7lol' and… | - |
| fs_3anha_14 | salam, law sama7t momken as2al 3anha, el se3er 3addeh fel 7aga di? | reject 3/5 | accept 4/5 | degeneracy: Garbled phrase 'el se3er 3addeh'.; fluency: 'el se3er 3addeh' garbl… | fluency: 'el se3er 3addeh' illogical when buyer asks; incoherent exchange. |
| fs_enha_14 | Momken a3raf enha bekam ya fendi, 3ayez ashteri el 7aga di mn 3andoko | reject 3/5 | accept 5/5 | anchor: 'a3raf enha bekam' -- 'that it' doesn't fit; reads like anhi 'which', a… | - |
| fs_3ard_00 | lissa wasalt el beit ba3d yom tawil w fi 3ard 7ilo 3ala el television… | reject 2/5 | accept 4/5 | degeneracy: Trailing garbled fragment '2alle3 el ta3ban'.; fluency: '2alle3 el … | fluency: '2alle3 el ta3ban' unidiomatic and unclear; sentence confusing. |
| fs_3ard_04 | el zahma el naharda magnoona w 3ard el share3 mosh kafi le kotor el 3… | reject 3/5 | accept 4/5 | degeneracy: Repeats 'share3 ... shari3' twice in one sentence.; fluency: 'le ko… | degeneracy: Redundant repetition: '3ard el share3' then 'fel shari3' repeats st… |
| fs_3arada_03 | ya sa7by el 3arada elli fel oda beti2el, momken tesa3dni 2ashilha 3al… | accept 4/5 | reject 3/5 | fluency: "beti2el" is wrong; Egyptian says "te2eela" (heavy). | anchor: A '3arada' in a room moved onto a car is strained; sense not clearly cr… |
| fs_3ashan_00 | ana lessa rag3 el beit delwa2ty ta3ban gedan mn el shoghl 3ashan ana … | reject 3/5 | accept 5/5 | coherence: Causal link reversed: tired from work 'because' I want to sleep.; fl… | - |
