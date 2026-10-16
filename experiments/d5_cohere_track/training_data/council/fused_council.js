export const meta = {
  name: 'fused-council',
  description: 'Five-lens council on fused training sentences; one judge per pending chunk and lens writes lens_chunk.json to a durable verdict dir',
  phases: [{ title: 'Judge', detail: 'one judge per pending job; model and effort come from args' }],
}

// Method of record for the fused-sentence council.
//
// Args come from council_jobs.py --council fused: {council, model, effort,
// chunkDir, outDir, sizes, jobs}. Policy: model opus / effort low for bulk
// fleets, opus / xhigh for calibration and audits. Only the pending jobs are
// listed, so a council that dies at a session limit resumes by re-running
// council_jobs.py and this script.
//
// LENSES, prompt and SUMMARY are byte-identical to the Opus council run of
// 2026-09-25 (session workflow council-fused-v2, run wf_c8f42ae0-f11), and a
// test pins their sha256: editing them makes every verdict dir judged so far
// incomparable with new ones. The prompt names no model, so a Sonnet and an
// Opus panel see exactly the same words.

const LENSES = {
  fluency: 'FLUENCY. Would an Egyptian actually type this sentence in a chat? Natural Egyptian word order, idioms and collocations; no word-for-word calques from English or MSA; Franco spelling within normal Egyptian habits.',
  anchor: 'ANCHOR SENSE. Is the anchor word used in exactly the sense its Arabic and gloss name? Would an Egyptian reader take the anchor as THAT Arabic word here, not as a look-alike word or a different sense? If the gloss lists several senses, any one of them is fine. Invalid if the sense is wrong, strained, or genuinely ambiguous. This sentence will train a model to answer "which Arabic word is <anchor> here", so a wrong sense is a wrong label.',
  degeneracy: 'DEGENERACY. Generation artifacts: a repeated word or phrase, filler, a trailing fragment, two sentences glued together, template or list shape, echoed instructions, or English words an Egyptian would not use there.',
  coherence: 'COHERENCE. Does the sentence make sense as a whole: one clear situation, logically consistent, nothing contradictory or nonsensical?',
  register: 'REGISTER. Everyday Egyptian colloquial: not MSA/fusha, not Levantine, Gulf or Maghrebi forms, not formal written style.',
}

const prompt = (chunk, lens, a) => `You are one of five independent judges on a council verifying synthetic training sentences for an Egyptian Franco-Arabic (Arabizi) fine-tuning dataset. Each item is ONE sentence an LLM wrote around an ANCHOR word, whose Arabic form and English gloss were confirmed earlier by a council.

Your lens is ${LENSES[lens]}
Judge ONLY this aspect -- four other judges cover the rest. Be strict: this data trains a model, and a wrong accept costs more than a wrong reject.

1. Read the chunk file ${a.chunkDir}/${chunk}.json -- a JSON array of items with keys sid, rid, anchor, arabic, gloss, sentence.
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

const COUNCIL = 'fused'
const fail = msg => {
  throw new Error(`fused council: ${msg} (generate args with council_jobs.py --council ${COUNCIL})`)
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
