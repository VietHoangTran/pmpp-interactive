#!/usr/bin/env node
// Generate narration takes through the ElevenLabs text-to-speech API, for when credits run
// low (the reader's decision, 2026-11-20). The usual path is the MCP speech tool with
// eleven_v4; this one calls the API directly, with the course's fixed voice and one of
// the models tools/narration.mjs accepts (default eleven_v4_turbo).
//
//   ELEVENLABS_API_KEY=... node tools/tts-api.mjs <chapter> <clip> [--model ID] [--takes N] [--out DIR]
//
// It writes DIR/<chapter>_<clip>_take<N>.mp3 and prints each take's character cost (the
// API's character-cost header). It does not choose or register a take: compare the takes
// with the script, then run
//   node tools/narration.mjs add <chapter> <clip> <file> --take N --model ID
//
// The key is read from the environment only, never from a file in the repository, and is
// never printed. A request that succeeded is never repeated; a refusal for concurrency
// (HTTP 429) started nothing and is retried after a wait.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODELS, VOICE } from './narration.mjs';

const REPO = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function arg(rest, name, fallback) {
  const i = rest.indexOf(name);
  return i >= 0 ? rest[i + 1] : fallback;
}

async function main() {
  const [chapter, clip, ...rest] = process.argv.slice(2);
  if (!chapter || !clip) {
    console.error('usage: ELEVENLABS_API_KEY=... node tools/tts-api.mjs <chapter> <clip> [--model ID] [--takes N] [--out DIR]');
    process.exit(2);
  }
  const key = process.env.ELEVENLABS_API_KEY;
  if (!key) throw new Error('ELEVENLABS_API_KEY is not set in the environment');
  const model = arg(rest, '--model', 'eleven_v4_turbo');
  if (!MODELS.includes(model)) throw new Error(`model ${model} is not one of ${MODELS.join(', ')}`);
  const takes = Number(arg(rest, '--takes', '2'));
  const out = arg(rest, '--out', path.join(os.tmpdir(), 'pmpp-narration'));
  fs.mkdirSync(out, { recursive: true });

  const scriptPath = path.join(REPO, 'content', chapter, 'narration', clip + '.txt');
  const text = fs.readFileSync(scriptPath, 'utf8').replace(/\r\n/g, '\n').trim();
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${VOICE}?output_format=mp3_44100_128`;

  for (let t = 1; t <= takes; t++) {
    let res;
    for (let attempt = 0; ; attempt++) {
      res = await fetch(url, {
        method: 'POST',
        headers: { 'xi-api-key': key, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_id: model }),
      });
      if (res.status !== 429 || attempt >= 10) break;
      await new Promise((r) => setTimeout(r, 20000)); // concurrency limit: nothing was started
    }
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`take ${t}: HTTP ${res.status} ${body.slice(0, 300)}`);
    }
    const file = path.join(out, `${chapter}_${clip}_take${t}.mp3`);
    fs.writeFileSync(file, Buffer.from(await res.arrayBuffer()));
    console.log(`${file}\tcharacter-cost ${res.headers.get('character-cost') ?? '?'}\tmodel ${model}`);
  }
}

main().catch((e) => {
  console.error(String(e.message ?? e));
  process.exit(1);
});
