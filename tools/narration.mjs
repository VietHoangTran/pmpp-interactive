#!/usr/bin/env node
// Narration bookkeeping. Each clip is a script (.txt) with its audio (.mp3) beside it:
//   content/<chapter>/narration/<clip>.txt
//   content/<chapter>/narration/<clip>.mp3
//   content/<chapter>/narration/manifest.json   clip -> { file, sha256 of the script, duration, ... }
//
//   node tools/narration.mjs status [chapter]            which clips are missing or stale
//   node tools/narration.mjs add <chapter> <clip> <audio-file> [--take N]
//                                                        register audio generated for the clip's current script
//
// Audio is generated only with the ElevenLabs MCP speech tool (model and voice below);
// this script never calls any API. A clip is STALE when its script changed after its
// audio was made, and only stale or missing clips need regenerating.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const REPO = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const CONTENT = path.join(REPO, 'content');
export const MODEL = 'eleven_v4';
export const VOICE = 'tnSpp4vdxKPjI9w0GnoV';

const dirOf = (chapter) => path.join(CONTENT, chapter, 'narration');
// Hash the script as it is sent to the speech tool: trimmed, with Unix line endings.
const normalise = (s) => s.replace(/\r\n/g, '\n').trim();
const hash = (s) => crypto.createHash('sha256').update(normalise(s)).digest('hex');

function readManifest(chapter) {
  const p = path.join(dirOf(chapter), 'manifest.json');
  if (!fs.existsSync(p)) return { model: MODEL, voice: VOICE, clips: {} };
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function writeManifest(chapter, m) {
  const clips = Object.fromEntries(Object.keys(m.clips).sort().map((k) => [k, m.clips[k]]));
  fs.writeFileSync(path.join(dirOf(chapter), 'manifest.json'), JSON.stringify({ model: m.model, voice: m.voice, clips }, null, 1) + '\n');
}

function duration(file) {
  try {
    const out = execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file], { encoding: 'utf8' });
    const d = Number(out.trim());
    return Number.isFinite(d) ? Math.round(d * 1000) / 1000 : null;
  } catch {
    return null; // ffprobe not installed: the player reads the duration from the file instead
  }
}

function status(only) {
  const chapters = fs.existsSync(CONTENT) ? fs.readdirSync(CONTENT).filter((c) => fs.existsSync(dirOf(c))) : [];
  let pending = 0;
  for (const chapter of chapters.sort()) {
    if (only && chapter !== only) continue;
    const m = readManifest(chapter);
    const scripts = fs.readdirSync(dirOf(chapter)).filter((f) => f.endsWith('.txt')).sort();
    for (const f of scripts) {
      const clip = f.replace(/\.txt$/, '');
      const text = fs.readFileSync(path.join(dirOf(chapter), f), 'utf8');
      const entry = m.clips[clip];
      let state = 'ok';
      if (!entry || !fs.existsSync(path.join(dirOf(chapter), entry.file))) state = 'MISSING';
      else if (entry.sha256 !== hash(text)) state = 'STALE';
      else if (m.model !== MODEL || m.voice !== VOICE) state = 'STALE (model or voice changed)';
      if (state !== 'ok') pending++;
      console.log(`${state.padEnd(8)} ${chapter}/${clip}  (${normalise(text).split(/\s+/).length} words${entry?.duration ? `, ${entry.duration}s` : ''})`);
    }
    for (const clip of Object.keys(m.clips))
      if (!scripts.includes(clip + '.txt')) console.log(`ORPHAN   ${chapter}/${clip}  (audio without a script)`);
  }
  console.log(pending ? `\n${pending} clip(s) need generating with model ${MODEL}, voice ${VOICE}.` : '\nAll narration is up to date.');
  return pending ? 1 : 0;
}

function add(chapter, clip, audio, take) {
  const script = path.join(dirOf(chapter), clip + '.txt');
  if (!fs.existsSync(script)) throw new Error(`no script at ${path.relative(REPO, script)}`);
  if (!fs.existsSync(audio)) throw new Error(`no audio file at ${audio}`);
  const dest = path.join(dirOf(chapter), clip + '.mp3');
  fs.copyFileSync(audio, dest);
  const m = readManifest(chapter);
  m.model = MODEL;
  m.voice = VOICE;
  m.clips[clip] = {
    file: clip + '.mp3',
    sha256: hash(fs.readFileSync(script, 'utf8')),
    duration: duration(dest),
    bytes: fs.statSync(dest).size,
    generated: new Date().toISOString().slice(0, 10),
    ...(take ? { take } : {}),
  };
  writeManifest(chapter, m);
  console.log(`registered ${chapter}/${clip}: ${m.clips[clip].duration ?? '?'}s, ${m.clips[clip].bytes} bytes`);
}

const [cmd, ...rest] = process.argv.slice(2);
if (cmd === 'status') process.exit(status(rest[0]));
else if (cmd === 'add' && rest.length >= 3) {
  const t = rest.indexOf('--take');
  add(rest[0], rest[1], rest[2], t >= 0 ? Number(rest[t + 1]) : undefined);
} else {
  console.error('usage: node tools/narration.mjs status [chapter] | add <chapter> <clip> <audio-file> [--take N]');
  process.exit(2);
}
