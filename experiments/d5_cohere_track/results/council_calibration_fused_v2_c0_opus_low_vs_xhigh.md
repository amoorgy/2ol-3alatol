# Council calibration: fused_v2_c0_opus_low_vs_xhigh

**opus/low rejects 22.6% of 115 fused items vs opus/xhigh's 42.6% (leniency delta +20.0 pp); 29.2% [20.8%, 39.4%] of opus/low's accepts are rejected by opus/xhigh.**

Reproduce:

```
python3 experiments/d5_cohere_track/training_data/council/compare_councils.py --council fused --chunk-dir data/generated/v2/council/fused_chunks --a-dir data/generated/v2/council/fused_votes --a-label opus/xhigh --b-dir data/generated/v2/council/fused_votes_opus_low --b-label opus/low --name fused_v2_c0_opus_low_vs_xhigh --chunks c0
```

## Config

| field | value |
|---|---|
| council | fused (lenses: anchor, coherence, degeneracy, fluency, register) |
| final verdict | ACCEPT at >= 4/5 lenses |
| reference A | opus/xhigh -- `data/generated/v2/council/fused_votes` (stamp: none) |
| candidate B | opus/low -- `data/generated/v2/council/fused_votes_opus_low` (stamp: yes) |
| corpus | `data/generated/v2/council/fused_chunks`: 1 chunks, 115 items (sha256 per chunk in the JSON) |
| seed | none: deterministic, no sampling |

## Final verdicts (>= 4/5)

| | opus/low accept | opus/low reject | total |
|---|---:|---:|---:|
| **opus/xhigh accept** | 63 | 3 | 66 |
| **opus/xhigh reject** | 26 | 23 | 49 |
| total | 89 | 26 | 115 |

| metric | value |
|---|---|
| reject rate opus/xhigh | 42.6% |
| reject rate opus/low | 22.6% |
| leniency delta (A reject - B reject) | +20.0 pp |
| agreement / Cohen's kappa | 74.8% / 0.451 |
| opus/low false-accept vs opus/xhigh (B accepts that A rejects) | 26/89 = 29.2% [20.8%, 39.4%] |
| opus/low miss rate vs opus/xhigh (A rejects that B accepts) | 26/49 = 53.1% [39.4%, 66.3%] |
| Fleiss' kappa across lenses, opus/xhigh / opus/low | 0.075 / 0.109 |

## Per lens

| lens | n | opus/xhigh accept | opus/low accept | leniency (pp) | Cohen's kappa | only opus/xhigh rejects | only opus/low rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor | 115 | 83.5% | 87.8% | +4.3 | 0.683 | 7 | 2 |
| coherence | 115 | 86.1% | 82.6% | -3.5 | 0.671 | 3 | 7 |
| degeneracy | 115 | 73.0% | 96.5% | +23.5 | 0.178 | 27 | 0 |
| fluency | 115 | 19.1% | 45.2% | +26.1 | 0.372 | 32 | 2 |
| register | 115 | 82.6% | 96.5% | +13.9 | 0.292 | 16 | 0 |

## opus/low at other thresholds (vs opus/xhigh at >= 4)

Calibration data only. A threshold picked from this table must be validated on a separate audit sample of the bulk run (`sample_audit.py`), not on these chunks.

| opus/low min agree | opus/low accept | agreement | kappa | opus/low false-accept vs opus/xhigh | opus/low miss rate |
|---:|---:|---:|---:|---|---|
| 1 | 100.0% | 57.4% | 0.000 | 49/115 = 42.6% [34.0%, 51.7%] | 49/49 = 100.0% [92.7%, 100.0%] |
| 2 | 99.1% | 58.3% | 0.023 | 48/114 = 42.1% [33.4%, 51.3%] | 48/49 = 98.0% [89.3%, 99.6%] |
| 3 | 87.8% | 69.6% | 0.315 | 35/101 = 34.6% [26.1%, 44.3%] | 35/49 = 71.4% [57.6%, 82.2%] |
| 4 | 77.4% | 74.8% | 0.451 | 26/89 = 29.2% [20.8%, 39.4%] | 26/49 = 53.1% [39.4%, 66.3%] |
| 5 | 44.4% | 76.5% | 0.538 | 6/51 = 11.8% [5.5%, 23.4%] | 6/49 = 12.2% [5.7%, 24.2%] |

## Final-verdict disagreements (29; first 29 shown, all in the JSON)

| sid | text | opus/xhigh | opus/low | opus/xhigh rejected on | opus/low rejected on |
|---|---|---|---|---|---|
| fs_2a2ema_02 | embare7 katabt 2a2ema gdeeda lel 7agat elli lazem ashtraha men el so2… | reject 3/5 | accept 4/5 | degeneracy: Garbled trailing fragment 'akammel shy el yom' tacked onto the end.… | fluency: ending 'akammel shy el yom' is garbled |
| fs_2a2em_09 | ya 3am shoghlok el gedid 2a2em 3ala el safar kteer, 3ashan keda magab… | reject 3/5 | accept 5/5 | fluency: 'magabelnash' is not a real form; '2a2em 3ala el safar' reads MSA.; re… | - |
| fs_2a2em_05 | el khotta 2a2em 3ala ra2y ommy, yalla nettaf2 ma3 el 3a2ila 3ala agaz… | reject 3/5 | accept 4/5 | fluency: Gender error: 'el khotta 2a2em' should be feminine (2ayma).; register:… | fluency: gender mismatch: khotta 2a2ema; phrasing unnatural |
| fs_dayra_03 | Momken tesa3edni fi wara2 el dayra 3ashan ana mesh 3aref a5alleso wah… | reject 3/5 | accept 5/5 | fluency: 'wara2 el dayra' is not an Egyptian collocation for department paperwo… | - |
| fs_2al_01 | bokra lazem nrouh el souq el sobh 3ashan ahmed 2al en el mawqe3 hayft… | reject 3/5 | accept 4/5 | fluency: 'el mawqe3 hayfta7' calque/odd for the market opening.; register: 'el … | fluency: 'el mawqe3 hayfta7' unnatural for a market |
| fs_2alb_11 | el match el naharda kan 3al 2alb awi w 3awez a3eedo tani ba3d ma el d… | reject 3/5 | accept 4/5 | anchor: 'kan 3al 2alb awi' is not an idiom; heart sense is strained and unclear… | fluency: '3al 2alb' and 'dawwarni' unidiomatic |
| fs_3abr_09 | ana b2ali zamaan ma shuftak, bas 3abr el-telefon 3reft akhbarak, ezay… | reject 2/5 | accept 5/5 | degeneracy: Repeats 'akhbarak' twice; formulaic greeting tacked onto end.; flue… | - |
| fs_3abr_08 | akolt koshary 3abr el shari3 embareh ma3 sa7by w el akl kan tayyeb awi | reject 2/5 | accept 4/5 | anchor: English calque 'across the street' as a location; Arabic 3abr doesn't m… | fluency: '3abr el shari3' wrong sense, unnatural |
| fs_3abr_12 | kol el ayyam bas7a badri w baghsel weshi w babos el shams 3abr el-sho… | accept 4/5 | reject 3/5 | fluency: '3abr el-shobak' formal calque; Egyptian says 'men el shebbak'. | coherence: Kissing the sun through window is odd; fluency: 'babos el shams' non… |
| fs_3ebra_06 | el khabar da asa2ny awi, bas akhodt menno 3ebra 7elwa fel 7ayah w ba2… | reject 2/5 | accept 4/5 | degeneracy: Garbled 'asa2ny' and incoherent trailing 'ba2a 3aref ra7ty'.; fluen… | fluency: 'ba2a 3aref ra7ty' incoherent |
| fs_3ebra_12 | kol youm basa7a 6:30 badry we da 3ebra en el nom el badry beyfara2 aw… | accept 4/5 | reject 3/5 | fluency: 'da 3ebra en el nom...' calques 'this is a lesson that'. | anchor: lesson sense strained; means proof/sign here; fluency: 'da 3ebra en' un… |
| fs_3adam_08 | el akl kan tayeb awi bas kan feh 3adam el mal7 fa ma3agabetneesh. | reject 2/5 | accept 4/5 | coherence: Says the food was very good, then that they didn't like it; self-con… | fluency: '3adam el mal7' calque; Egyptians say mafihoosh mal7 |
| fs_2adm_09 | wallahy we7ashteny ya 3am 2adm el zaman ma shoftak, 2olly akhbarak eh | reject 3/5 | accept 4/5 | fluency: '2adm el zaman ma shoftak' not an Egyptian idiom; 'men zaman' expected… | fluency: '2adm el zaman' not Egyptian usage |
| fs_2adm_12 | el sob7 ba2oom men el noom, 2aghsel 2adm el yemeen we ba3den 2ashrab … | reject 3/5 | accept 4/5 | degeneracy: Trailing '3ala el-fatra' is a garbled, meaningless fragment.; fluen… | fluency: washing '2adm el yemeen' unnatural; garbled |
| fs_3ala2a_06 | ana lissa sama3t el khabar da w mesh fahem eh el 3ala2a baina el 7aga… | reject 3/5 | accept 4/5 | fluency: 'baina' is MSA; Egyptian chat uses 'ben'.; register: 'baina' is the MS… | register: "baina" is MSA form; Egyptian colloquial uses bein |
| fs_3alya_11 | el match el bare7 kan gamed awi, w nasbet el hodour fel stadium 3alya… | reject 2/5 | accept 5/5 | degeneracy: English 'stadium' unnatural here; an Egyptian would say 'el estad'.… | - |
| fs_3alya_02 | embare7 lama shreit el kotob, la2et el se3r 3alya awi we mal2etsh flu… | reject 2/5 | accept 4/5 | coherence: Says they bought the books yet lacked enough money to pay; contradic… | fluency: se3r masculine; should be 3aly |
| fs_3alena_14 | ya 3am el 7aga di be kam 3alena, 2olli 3ad3ad el takhfid men fadlak | reject 3/5 | accept 4/5 | degeneracy: Reduplicated non-word '3ad3ad' is a generation artifact.; fluency: … | fluency: '3ad3ad el takhfid' nonsense |
| fs_3alena_06 | El khabar da sa3b awi w gamed, w 3alena nshof 7al be5alas el fatra el… | accept 4/5 | reject 3/5 | fluency: 'sa3b awi w gamed' contradictory; 'be5alas el fatra' garbled. | coherence: News both hard and awesome, contradictory; fluency: 'be5alas el fatr… |
| fs_lena_09 | fatra taweela ma shoftak, lena 3andena keteer kalemat w akhbar 3ayzee… | reject 1/5 | accept 4/5 | anchor: 'lena 3andena' doubles possession; lena's role is garbled and strained.… | fluency: 'lena 3andena' redundant, unnatural |
| fs_3amal_06 | wallahi el khabar da gamed bas el 3amal da mesh sahl khales we lazem … | reject 2/5 | accept 4/5 | degeneracy: Garbled non-word 'nolkaf'.; fluency: 'nolkaf' not a word; '7ata' MS… | fluency: 'nolkaf' nonsense; MSA 3amal |
| fs_3amal_14 | el se3r beta3 el 3amal elly hena kam, 7ad y2olli 3al 7aga di? | reject 3/5 | accept 4/5 | anchor: Shopping 'se3r el 3amal elly hena' -- unclear what 3amal denotes; strai… | fluency: 'se3r el 3amal' unnatural |
| fs_ana_12 | kol youm ana ba2oom badri w ashrab 2ahwa w bastahamma w 2afta7 3ala t… | reject 3/5 | accept 4/5 | degeneracy: List-shaped w-chain; garbled '2afta7 3ala tol 3ala el ftoor'.; flue… | fluency: 'bastahamma w 2afta7 3ala el ftoor' garbled |
| fs_3andha_00 | ana lissa wasalt el beit ba3d yom ta3ban awy w sa7bety 3andha moshkel… | reject 3/5 | accept 5/5 | degeneracy: Three unrelated clauses glued together with 'w'.; fluency: 'yom ta3… | - |
| fs_3andoh_00 | ana lissa wasalt el beit ba3d yom taweel we 3ayyan awi, 3andoh el wa2… | reject 2/5 | accept 4/5 | coherence: Unintroduced 'he has time to listen and yirawwini' doesn't connect t… | fluency: '3andoh' referent missing; incoherent |
| fs_3anha_02 | Sa2alni 3anha embare7 fel share3 bas ma 3reftesh ashra7lol 3ashan kan… | reject 3/5 | accept 4/5 | degeneracy: Garbled tokens 'ashra7lol' and 'mad3oo3'.; fluency: 'ashra7lol' and… | fluency: 'ashra7lol', 'mad3oo3' garbled |
| fs_3anha_14 | salam, law sama7t momken as2al 3anha, el se3er 3addeh fel 7aga di? | reject 3/5 | accept 4/5 | degeneracy: Garbled phrase 'el se3er 3addeh'.; fluency: 'el se3er 3addeh' garbl… | fluency: 'as2al 3anha' unnatural for price |
| fs_enha_14 | Momken a3raf enha bekam ya fendi, 3ayez ashteri el 7aga di mn 3andoko | reject 3/5 | accept 4/5 | anchor: 'a3raf enha bekam' -- 'that it' doesn't fit; reads like anhi 'which', a… | fluency: 'a3raf enha bekam' unnatural |
| fs_3ard_04 | el zahma el naharda magnoona w 3ard el share3 mosh kafi le kotor el 3… | reject 3/5 | accept 5/5 | degeneracy: Repeats 'share3 ... shari3' twice in one sentence.; fluency: 'le ko… | - |
