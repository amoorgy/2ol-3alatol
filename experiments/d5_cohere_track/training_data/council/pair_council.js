export const meta = {
  name: 'pair-council',
  description: 'Five-lens council on flagged WORD to FORM substitution pairs; one judge per pending chunk and lens writes lens_chunk.json to a durable verdict dir',
  phases: [{ title: 'Judge', detail: 'one judge per pending job; model and effort come from args' }],
}

// Method of record for the substitution-pair council.
//
// Items come from prep_council_pairs_v2.py: {sid, rid, word, form,
// why_flagged, examples: [{base, respelled}]}, sid == rid == "word->form".
// Each is ONE respelling judged once; its verdict applies to every training
// row carrying that pair, and emit_v2.py holds unjudged pairs back.
//
// Args come from council_jobs.py --council pairs: {council, model, effort,
// chunkDir, outDir, sizes, jobs}. Policy: model opus / effort low for bulk
// fleets, opus / xhigh for calibration and audits. The prompt names no model,
// so both panels see exactly the same words. The header, steps and SUMMARY
// follow the fused council's wording so the two councils differ only in
// what they judge.

const LENSES = {
  coherence: 'COHERENCE. With FORM in place, does each respelled sentence still say what its base sentence says? Read every other changed word as its base-sentence counterpart. Invalid if FORM changes, blurs or breaks the meaning of any example.',
  identity: 'IDENTITY. Would an Egyptian reading FORM in these sentences still take it as the same word -- the same Arabic -- as WORD? Invalid if FORM would be read as a different Egyptian word, or as an English word, in any example.',
  plausibility: 'PLAUSIBILITY. Would a real Egyptian actually type FORM for WORD in a chat? Invalid if no Egyptian would spell this word that way, even if a reader could still work out what was meant.',
  register: 'REGISTER. Is FORM consistent with Egyptian Franco-Arabic conventions: the usual digits for Arabic letters, Egyptian vowel habits, Egyptian pronunciation? Invalid if it follows another dialect, MSA, or no Franco convention at all.',
  skeptic: 'SKEPTIC. Actively look for a collision or ambiguity: an Egyptian word, a common Franco spelling of a different word, or an English word that FORM could be read as. Default to v = false if a plausible confusion exists; v = true only if you looked and found none.',
}

const prompt = (chunk, lens, a) => `You are one of five independent judges on a council verifying spelling substitutions for an Egyptian Franco-Arabic (Arabizi) fine-tuning dataset. Each item is ONE respelling, WORD -> FORM (keys word and form), shown in up to two real sentences as a base sentence and its respelled version. Accepting an item trains a model that FORM is the same word as WORD.

Judge ONLY that one pair. Other words in a respelled sentence may also be respelled; they are judged separately, so do not let them affect your verdict.

A native speaker ruled that writing ع as a vowel at the START of a word (3amr -> amr, 3and -> aand) and dropping an inner vowel (leha -> lha) are real Egyptian habits, and that dropping a vowel that sits next to another vowel (radio -> rado) usually reads fine in context too. Every pair here is one of the two harder cases -- a word-initial ع written as a vowel, or a vowel dropped beside another vowel -- flagged only because no rule can prove a specific pair safe (3nd -> and lands on an English word; 3amalia -> 3amala could read as a different word); why_flagged names the case and is not evidence against the pair.

Your lens is ${LENSES[lens]}
Judge ONLY this aspect -- four other judges cover the rest. Be strict: this data trains a model, and a wrong accept costs more than a wrong reject.

1. Read the chunk file ${a.chunkDir}/${chunk}.json -- a JSON array of items with keys sid, rid, word, form, why_flagged, examples (a list of {base, respelled} sentence pairs).
2. For EVERY item decide v = true (passes your lens) or v = false. "why" is required when v is false (at most 20 words) and may be "" when true.
3. Use the Write tool to write the complete result to ${a.outDir}/${lens}_${chunk}.json as a JSON array: one object per item, in the same order, with exactly the keys sid, rid, v, why (copy sid and rid unchanged). Skip no item and add nothing else to the file.

Finally return chunk, lens, the number of items you judged, and how many you marked false.`

const SUMMARY = {
  type: 'object',
  properties: {
    chunk: { type: 'string' }, lens: { type: 'string' },
    n_judged: { type: 'integer' }, n_invalid: { type: 'integer' },
  },
  required: ['chunk', 'lens', 'n_judged', 'n_invalid'],
}

const COUNCIL = 'pairs'
const fail = msg => {
  throw new Error(`pairs council: ${msg} (generate args with council_jobs.py --council ${COUNCIL})`)
}
const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key)
if (!args || typeof args !== 'object') fail('args are required')
if (args.council !== COUNCIL) fail(`args.council is ${args.council}, expected ${COUNCIL}`)
for (const key of ['model', 'effort', 'chunkDir', 'outDir', 'sizes']) {
  if (!args[key]) fail(`args.${key} is required`)
}
if (!Array.isArray(args.jobs)) fail('args.jobs must be a list of {chunk, lens}')
const bad = args.jobs.filter(j => !j || !own(LENSES, j.lens) || !own(args.sizes, j.chunk))
if (bad.length) fail(`${bad.length} jobs name an unknown lens or chunk, e.g. ${JSON.stringify(bad[0])}`)

log(`${args.jobs.length} ${COUNCIL} judges pending on ${args.model}/${args.effort}`)
phase('Judge')
const results = await parallel(args.jobs.map(j => () =>
  agent(prompt(j.chunk, j.lens, args), {
    label: `${j.lens}:${j.chunk}`, phase: 'Judge', schema: SUMMARY,
    model: args.model, effort: args.effort, agentType: 'general-purpose',
  })))
const done = results.filter(Boolean)
const short = done.filter(r => r.n_judged !== args.sizes[r.chunk])
log(`${done.length}/${args.jobs.length} judges returned; ${short.length} judged a different count than the chunk holds`)
return {
  council: COUNCIL, model: args.model, effort: args.effort,
  returned: done.length, expected: args.jobs.length, short, results: done,
}
