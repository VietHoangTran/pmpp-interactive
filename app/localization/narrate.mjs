// Generate Vietnamese narration locally with Piper, then encode AAC on macOS.
import ts from 'typescript-parser';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const chapters = new URL('../src/chapters/', import.meta.url);
const content = new URL('../../content/', import.meta.url);
const [, , modelFile, ...chapterIds] = process.argv;
if (!modelFile) throw new Error('Usage: node localization/narrate.mjs <Piper model .onnx> [chapter IDs...]');
const selected = new Set(chapterIds);
const python = process.env.PMPP_PIPER_PYTHON ?? 'python3';
const piperModel = resolve(modelFile);
const voice = basename(piperModel, '.onnx');
const model = 'Piper';
const modelSha256 = createHash('sha256').update(readFileSync(piperModel)).digest('hex');
const configSha256 = createHash('sha256').update(readFileSync(piperModel + '.json')).digest('hex');
const items = new Map();

function property(node, name) {
  if (!node || !ts.isObjectLiteralExpression(node)) return undefined;
  return node.properties.find((p) => ts.isPropertyAssignment(p) && p.name.getText() === name)?.initializer;
}

for (const filename of readdirSync(chapters, { recursive: true }).filter((file) => file.endsWith('.tsx'))) {
  const chapter = filename.split('/')[0];
  if (selected.size && !selected.has(chapter)) continue;
  const source = readFileSync(new URL(filename, chapters), 'utf8');
  const file = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  function visit(node) {
    if (ts.isObjectLiteralExpression(node)) {
      const clip = property(node, 'clip');
      const caption = property(property(node, 'vietnamese'), 'caption');
      if (clip && caption && ts.isStringLiteralLike(clip) && ts.isStringLiteralLike(caption)) {
        if (!items.has(chapter)) items.set(chapter, new Map());
        items.get(chapter).set(clip.text, caption.text);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(file);
}

if (items.size === 0) throw new Error('No Vietnamese captions found.');
const work = mkdtempSync(join(tmpdir(), 'pmpp-vi-voice-'));
try {
  for (const [chapter, captions] of items) {
    const folder = new URL(chapter + '/narration/', content);
    const manifestPath = new URL('manifest.vi.json', folder);
    const previous = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, 'utf8')) : undefined;
    const clips = {};
    for (const [id, caption] of captions) {
      const sha256 = createHash('sha256').update(caption).digest('hex');
      const filename = id + '.vi.m4a';
      const output = fileURLToPath(new URL(filename, folder));
      const saved = previous?.clips[id];
      if (previous?.model === model && previous?.modelSha256 === modelSha256 && previous?.configSha256 === configSha256 &&
          saved?.sha256 === sha256 && saved.file === filename && existsSync(output)) {
        clips[id] = saved;
        continue;
      }
      const tempFile = join(work, chapter + '-' + id);
      const speech = tempFile + '.txt';
      const pcm = tempFile + '.wav';
      const encoded = tempFile + '.m4a';
      const script = caption.replace(/\bCPU\b/g, 'xê pê u').replace(/\bGPU\b/g, 'giê pê u');
      writeFileSync(speech, script);
      execFileSync(python, ['-m', 'piper', '-m', piperModel, '-f', pcm, '--input-file', speech]);
      execFileSync('afconvert', ['-f', 'm4af', '-d', 'aac', '-b', '64000', pcm, encoded]);
      const report = execFileSync('afinfo', [encoded], { encoding: 'utf8' });
      const duration = Number(/estimated duration: ([\d.]+) sec/.exec(report)?.[1]);
      if (!(duration > 0)) throw new Error('Missing duration for ' + id);
      writeFileSync(output, readFileSync(encoded));
      clips[id] = { file: filename, sha256, duration };
      console.log(chapter + '/' + id, duration.toFixed(2) + 's');
    }
    writeFileSync(manifestPath, JSON.stringify({ model, voice, modelSha256, configSha256, clips }, null, 2) + '\n');
  }
} finally {
  rmSync(work, { recursive: true, force: true });
}
