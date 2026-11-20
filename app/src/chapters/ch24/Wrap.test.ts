import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { numericOk, type ChoiceQ, type NumericQ } from '../../components/Ask';
import { VENDOR, threshold, tileFor } from '../../sim/hwtable';
import { DECODE, GAP, HGX, MI355X, questionsWrap } from './Wrap';

// The summary page shows no kernel (chapter 24 has no code), so there is no snippet to
// tie it to. This test pins the numbers its questions state to the model the start page
// runs on and to the vendor quotes in the facts file, and a few of its statements to
// the notes they come from.
const repo = path.resolve(__dirname, '../../../..');
const notes = fs.readFileSync(path.join(repo, 'notes/ch24.md'), 'utf8');
const facts = fs.readFileSync(path.join(repo, 'notes/today/ch24-facts.md'), 'utf8');
const page = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const correct = (q: ChoiceQ) => q.options.findIndex((o) => o.correct);
const numeric = (id: string) => questionsWrap[id] as NumericQ;

describe('summary page of chapter 24', () => {
  it('has four questions with ids under ch24/wrap, each with one way to be right', () => {
    const ids = Object.keys(questionsWrap);
    expect(ids).toEqual(['ch24/wrap/q-gap', 'ch24/wrap/q-perboard', 'ch24/wrap/q-mi355x', 'ch24/wrap/q-next']);
    for (const id of ids) {
      const q = questionsWrap[id];
      expect(q.id).toBe(id);
      expect(q.source).toBeTruthy();
      if (q.kind === 'choice') {
        expect(q.options.filter((o) => o.correct)).toHaveLength(1);
        expect(q.options.filter((o) => !o.correct).every((o) => !!o.why)).toBe(true);
      }
    }
  });

  it('q-gap: device memory over the host link is 125 in the 2025 column and 10.8 in 2007', () => {
    expect(GAP.y2025).toBe(125);
    expect(GAP.y2007).toBeCloseTo(10.8, 10);
    // On NVIDIA's figures for one board each: 76.8 / 4.0 and 8000 / 63.0.
    expect(GAP.vendor2007).toBeCloseTo(19.2, 10);
    expect(Math.round(GAP.vendor2025)).toBe(127);
    const q = numeric('ch24/wrap/q-gap');
    expect(numericOk(q, 125)).toBe(true);
    expect(numericOk(q, 10.8)).toBe(false);
    expect(numericOk(q, 93)).toBe(false);
  });

  it("q-perboard: the HGX board's FP32 over 8 is NVIDIA's per-GPU figure, and the per-GPU threshold is 9.375", () => {
    expect(HGX.boardFp32Tflops / HGX.gpus).toBe(VENDOR.b200.fp32.value);
    expect(threshold((HGX.boardFp32Tflops / HGX.gpus) * 1000, HGX.gpuBandwidthTBs * 1000)).toBe(9.375);
    expect(facts).toContain('FP32 | 600 TFLOPS');
    expect(facts).toContain('GPU Bandwidth | Up to 8TB/s');
    expect(correct(questionsWrap['ch24/wrap/q-perboard'] as ChoiceQ)).toBe(1);
  });

  it("q-mi355x: AMD's figures are quoted as the facts file has them; the threshold is 19.66 and T = 79", () => {
    expect(facts).toContain(`Peak Single Precision (FP32) Performance | ${MI355X.fp32Tflops} TFLOPs`);
    expect(facts).toContain(`Peak Single Precision Matrix (FP32) Performance | ${MI355X.fp32MatrixTflops} TFLOPs`);
    expect(facts).toContain(`Peak Half Precision Matrix (FP16) Performance | ${MI355X.fp16MatrixPflops} PFLOPs`);
    expect(facts).toContain(`Peak Memory Bandwidth | ${MI355X.bandwidthTBs} TB/s`);
    expect(facts).toContain('TF32 support through software emulation');
    const thr = threshold(MI355X.fp32Tflops * 1000, MI355X.bandwidthTBs * 1000);
    expect(thr).toBeCloseTo(19.6625, 10);
    expect(tileFor(thr)).toBe(79);
    const q = numeric('ch24/wrap/q-mi355x');
    expect(numericOk(q, 19.7)).toBe(true);
    expect(numericOk(q, 312.5)).toBe(false); // the FP16 matrix figure
  });

  it('q-next: the ratio of batch-1 decode doubles at 8 bits and doubles again at 4', () => {
    expect(DECODE.b16).toBeCloseTo(1, 2);
    expect(DECODE.b8 / DECODE.b16).toBeCloseTo(2, 10);
    expect(DECODE.b4 / DECODE.b16).toBeCloseTo(4, 10);
    expect(correct(questionsWrap['ch24/wrap/q-next'] as ChoiceQ)).toBe(2);
  });

  it('lists the errata of the notes and only uses the printed label inside that list', () => {
    for (const s of ['638x', '1,800 TB/s', 'a 18-year comparison', 'massively computing devices', 'the answer is a definitely no'])
      expect(notes.includes(s) || notes.includes(s.replace('x', '×')), s).toBe(true);
    for (const s of ['638x', '"1,800 TB/s"', '"a 18-year comparison"', '"massively computing devices"', '"the answer is a definitely no"'])
      expect(page, s).toContain(s);
    const errata = page.slice(page.indexOf('<ul data-errata>'), page.indexOf('</ul>', page.indexOf('<ul data-errata>')));
    expect(page.split('Tensor Core FP32').length - 1).toBe(1);
    expect(errata).toContain('Tensor Core FP32');
  });

  it('words the dated and secondary claims as the facts file does', () => {
    // A7: the first quarter of Blackwell volume sales ended on January 26, 2025.
    expect(facts).toContain('quarter ended January 26, 2025');
    expect(page).toContain('first quarter of Blackwell volume sales ended in January 2025');
    expect(page).not.toContain('not launch years');
    // A8, A11: the book's 80 and 192 match NVIDIA figures; where the book took them from is unknown.
    expect(page).toContain("80 TFLOPS matches NVIDIA's GB200 NVL72 rack figure divided by 72");
    expect(page).not.toMatch(/80 TFLOPS is NVIDIA's/);
    // A3: 86.4 GB/s was found only in a forum post, so the page does not say it appears only there.
    expect(facts).toContain('found the 86.4 number itself only in a 2007 post');
    expect(page).not.toContain('appears only in');
    // A5: PCIe rates come from the Linux kernel, not from NVIDIA.
    expect(page).toContain("PCIe rates from the Linux kernel's table, facts A5");
  });

  it('says nothing is being written and names the exercises as none', () => {
    expect(notes).toContain('The chapter has no exercises.');
    expect(page).not.toMatch(/being written/i);
  });
});
