import { existsSync, readFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { lessonMessages } from './messages.mjs';

const app = fileURLToPath(new URL('..', import.meta.url));
const query = '?pmpp-language=vi';
const escapeTemplate = (chunk) => chunk.replace(/[\\`]/g, '\\$&').replace(/\$\{/g, '\\${');
const escapeJsx = (text) => text.replace(/[&<>{}]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '{': '&#123;', '}': '&#125;',
})[char]);

function replacement(message, value) {
  if (message.kind === 'template') {
    const slots = [...value.matchAll(/\{(\d+)\}/g)].map((match) => Number(match[1]));
    if (slots.length !== message.expressions.length || slots.some((slot, index) => slot !== index)) return null;
    const parts = value.split(/\{(\d+)\}/g);
    const text = parts.map((part, i) => i % 2 ? '\${' + message.expressions[Number(part)] + '}' : escapeTemplate(part)).join('');
    return '`' + (/^\s/.test(message.text ?? '') ? ' ' : '') + text + (/\s$/.test(message.text ?? '') ? ' ' : '') + '`';
  }
  if (message.kind === 'jsx') {
    const prefix = /^\s*/.exec(message.text)[0];
    const suffix = /\s*$/.exec(message.text)[0];
    return prefix + escapeJsx(value) + suffix;
  }
  const text = (/^\s/.test(message.text) ? ' ' : '') + value + (/\s$/.test(message.text) ? ' ' : '');
  return message.kind === 'attribute' ? '{' + JSON.stringify(text) + '}' : JSON.stringify(text);
}

/** Vietnamese lesson modules use the same components, logic and keys as the English source. */
export function vietnameseLessons() {
  const catalogs = new Map();
  return {
    name: 'pmpp-vietnamese-lessons',
    enforce: 'pre',
    transform(source, id) {
      if (!id.endsWith(query)) return null;
      const file = id.slice(0, -query.length);
      const lesson = /^src\/chapters\/([^/]+)\/([^/]+)\.tsx$/.exec(relative(app, file).replaceAll('\\', '/'));
      if (!lesson) return null;
      const [, chapter, section] = lesson;
      const catalog = resolve(app, 'localization/vi', chapter, section + '.json');
      if (!catalogs.has(catalog)) {
        if (existsSync(catalog)) {
          this.addWatchFile(catalog);
          catalogs.set(catalog, JSON.parse(readFileSync(catalog, 'utf8')));
        } else catalogs.set(catalog, {});
      }
      const strings = catalogs.get(catalog);
      const messages = lessonMessages(source, file);
      const edits = [];
      for (const message of messages) {
        if (!Object.hasOwn(strings, message.key)) continue;
        const translation = replacement(message, strings[message.key]);
        if (translation != null) edits.push({ ...message, translation });
      }
      let code = source;
      let end = source.length;
      for (const edit of edits.sort((a, b) => b.start - a.start)) {
        if (edit.end > end) continue;
        code = code.slice(0, edit.start) + edit.translation + code.slice(edit.end);
        end = edit.start;
      }
      return { code, map: null };
    },
    watchChange(id) {
      if (id.includes('/localization/vi/')) catalogs.clear();
    },
  };
}
