import { readFileSync, readdirSync } from 'node:fs';
import { lessonMessages } from './messages.mjs';

const translations = new URL('./vi/', import.meta.url);
const chapters = new URL('../src/chapters/', import.meta.url);
let total = 0;
for (const path of readdirSync(translations, { recursive: true }).filter((file) => file.endsWith('.json'))) {
  const source = new URL(path.replace(/\.json$/, '.tsx'), chapters);
  const messages = lessonMessages(readFileSync(source, 'utf8'), source.pathname);
  const catalog = JSON.parse(readFileSync(new URL(path, translations), 'utf8'));
  const keys = new Set(messages.map(({ key }) => key));
  const missing = [...keys].filter((key) => !Object.hasOwn(catalog, key));
  const stale = Object.keys(catalog).filter((key) => !keys.has(key));
  const invalid = messages.filter(({ key, expressions }) => {
    const value = catalog[key];
    if (typeof value !== 'string' || !value.trim()) return true;
    const slots = [...value.matchAll(/\{(\d+)\}/g)].map((match) => Number(match[1]));
    return slots.length !== (expressions?.length ?? 0) || slots.some((slot, index) => slot !== index);
  });
  console.log(`${path}: ${keys.size - missing.length}/${keys.size} strings`);
  if (missing.length || stale.length || invalid.length) {
    console.error({ missing, stale, invalid: invalid.map(({ key }) => key) });
    process.exitCode = 1;
  }
  total += keys.size;
}
console.log(`${total} catalogued messages`);

let sections = 0;
let localizedSections = 0;
for (const directory of readdirSync(chapters, { withFileTypes: true })) {
  if (!directory.isDirectory()) continue;
  const index = readFileSync(new URL(directory.name + '/index.tsx', chapters), 'utf8');
  sections += [...index.matchAll(/^\s*load: /gm)].length;
  localizedSections += [...index.matchAll(/^\s*vietnamese: /gm)].length;
}
console.log(`${localizedSections}/${sections} lesson sections offer Vietnamese.`);
