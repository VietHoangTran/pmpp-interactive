// Cutting one function out of a C-family source file (.cu, .metal, .mm, .cpp), so a
// lesson can show the real, tested lab file and only the part it is about.
//
// It is a small scanner, not a parser: it knows comments, string and character
// literals and brace nesting, which is enough for the lab files. A definition is the
// name followed by a parameter list and then, before any `;`, `,`, `)` or `=`, an
// opening brace. Prototypes, calls and kernel launches (`name<<<...>>>(...)`) are skipped.

export interface Excerpt {
  /** The function's lines, from its leading comment (if kept) to its closing brace. */
  text: string;
  /** 1-based line of the first excerpted line in the file, for the code view's numbering. */
  firstLine: number;
  /** 1-based line of the closing brace. */
  lastLine: number;
}

export interface ExcerptOptions {
  /** Keep the comment lines directly above the function (default true). */
  comment?: boolean;
}

/** Marks every character that is code (not inside a comment or a literal). */
function codeMask(src: string): boolean[] {
  const mask = new Array<boolean>(src.length).fill(true);
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    const n = src[i + 1];
    if (c === '/' && n === '/') {
      while (i < src.length && src[i] !== '\n') mask[i++] = false;
    } else if (c === '/' && n === '*') {
      const end = src.indexOf('*/', i + 2);
      const stop = end < 0 ? src.length : end + 2;
      while (i < stop) mask[i++] = false;
    } else if (c === '"' || c === "'") {
      mask[i++] = false;
      while (i < src.length && src[i] !== c && src[i] !== '\n') {
        if (src[i] === '\\') mask[i++] = false;
        mask[i++] = false;
      }
      if (i < src.length) mask[i++] = false;
    } else i++;
  }
  return mask;
}

const lineOf = (src: string, at: number) => src.slice(0, at).split('\n').length;

/** The definition of function `name` in `src`, or null when the file has none. */
export function excerptFunction(src: string, name: string, opts: ExcerptOptions = {}): Excerpt | null {
  const keepComment = opts.comment ?? true;
  const mask = codeMask(src);
  const word = new RegExp(`(?<![\\w$])${name.replace(/[$]/g, '\\$')}(?![\\w$])`, 'g');
  for (let m = word.exec(src); m; m = word.exec(src)) {
    if (!mask[m.index]) continue;
    // A member access or a qualified call through an object is not a definition.
    const before = src.slice(0, m.index).replace(/\s+$/, '');
    if (before.endsWith('.') || before.endsWith('->')) continue;
    // Template arguments (name<T>) are allowed; the parameter list must follow.
    let i = m.index + name.length;
    const skipSpace = () => {
      while (i < src.length && (/\s/.test(src[i]) || !mask[i])) i++;
    };
    skipSpace();
    if (src[i] === '<') {
      if (src.startsWith('<<<', i)) continue; // a kernel launch
      let depth = 0;
      for (; i < src.length; i++) {
        if (!mask[i]) continue;
        if (src[i] === '<') depth++;
        else if (src[i] === '>' && --depth === 0) break;
        else if (src[i] === ';' || src[i] === '{') break;
      }
      if (src[i] !== '>') continue;
      i++;
      skipSpace();
    }
    if (src[i] !== '(') continue;
    // The parameter list.
    let depth = 0;
    for (; i < src.length; i++) {
      if (!mask[i]) continue;
      if (src[i] === '(') depth++;
      else if (src[i] === ')' && --depth === 0) break;
    }
    if (depth !== 0) continue;
    i++;
    // Between the parameters and the body: qualifiers, attributes, a trailing return type.
    let body = -1;
    let attr = 0;
    for (; i < src.length; i++) {
      if (!mask[i]) continue;
      const c = src[i];
      if (c === '[' || c === '(') attr++;
      else if (c === ']' || c === ')') {
        if (--attr < 0) break;
      } else if (attr > 0) continue;
      else if (c === '{') {
        body = i;
        break;
      } else if (c === ';' || c === ',' || c === '=' || c === '?' || c === '}') break;
    }
    if (body < 0) continue;
    // The matching closing brace.
    depth = 0;
    let end = -1;
    for (let j = body; j < src.length; j++) {
      if (!mask[j]) continue;
      if (src[j] === '{') depth++;
      else if (src[j] === '}' && --depth === 0) {
        end = j;
        break;
      }
    }
    if (end < 0) return null;
    const lines = src.split('\n');
    // The declaration starts on the line with the name, or above it when the return type,
    // template header or attributes sit on lines of their own.
    let first = lineOf(src, m.index);
    while (first > 1) {
      const prev = lines[first - 2].trim();
      if (!prev || prev.startsWith('//') || prev.startsWith('/*') || prev.startsWith('*') || prev.startsWith('#')) break;
      if (/[;{}]$/.test(prev) || prev.endsWith('*/')) break;
      first--;
    }
    if (keepComment) {
      while (first > 1) {
        const prev = lines[first - 2].trim();
        if (prev.startsWith('//') || prev.startsWith('/*') || prev.startsWith('*')) first--;
        else break;
      }
    }
    const last = lineOf(src, end);
    return { text: lines.slice(first - 1, last).join('\n') + '\n', firstLine: first, lastLine: last };
  }
  return null;
}
