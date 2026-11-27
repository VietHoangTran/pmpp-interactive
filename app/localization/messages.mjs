import ts from 'typescript-parser';

const jsxAttributes = new Set([
  'aria-label', 'aria-description', 'alt', 'title', 'placeholder',
  'label', 'task', 'foot', 'prompt', 'unit', 'caption', 'description', 'checked', 'aria',
]);
const displayFields = new Set([
  'title', 'lede', 'label', 'why', 'explain', 'prompt', 'unit', 'caption',
  'description', 'detail', 'hint', 'summary', 'text', 'head', 'note', 'reason',
]);
const codeElements = new Set(['code', 'pre', 'kbd', 'samp', 'script', 'style']);
const compact = (text) => text.replace(/\s+/g, ' ').trim();

function isProse(text) {
  const words = text.match(/[a-zA-Z][a-zA-Z'-]*/g);
  if (!words || text.length > 4000 || !/^[\x20-\x7e\s\p{P}\p{S}\u00a0-\u024f]+$/u.test(text)) return false;
  if (/^(?:https?:|\.\.?\/|[\/#]|[a-z0-9_-]+\/)/i.test(text) || /^[-\w]+\.(?:cu|mm|h|cpp|md|txt|json|metal)$/i.test(text)) return false;
  if (words.length === 1) return /^(?:[A-Za-z][a-zA-Z-]+[.!?]?|[Cc]hapter \d+|[Ss]ection \d+\.\d+)$/.test(text);
  return words.some((word) => /[a-z]{2}/.test(word));
}

function displayed(node, context) {
  const parent = node.parent;
  if (ts.isJsxAttribute(parent)) return jsxAttributes.has(parent.name.getText());
  if (ts.isPropertyAssignment(parent)) return displayFields.has(parent.name.getText().replace(/^['"]|['"]$/g, ''));
  if (ts.isBinaryExpression(parent) && ![ts.SyntaxKind.PlusToken, ts.SyntaxKind.PlusEqualsToken].includes(parent.operatorToken.kind)) return false;
  if (ts.isElementAccessExpression(parent) || ts.isCaseClause(parent) || ts.isTypeNode(parent)) return false;
  if (context) return true;
  // Standalone function return values are display text only when they are sentences.
  return /[.!?]$/.test(node.text?.trim() ?? '') && /\s/.test(node.text ?? '') &&
    (ts.isReturnStatement(parent) || ts.isConditionalExpression(parent) || ts.isArrowFunction(parent));
}

/**
 * Extract only presentation text from a TypeScript lesson. Use the same extraction
 * when applying the catalog so stale entries cannot touch code, identifiers or routes.
 */
export function lessonMessages(source, filename) {
  const file = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  const messages = [];
  const add = (node, text, kind) => {
    const key = compact(text);
    if (!isProse(key)) return;
    messages.push({ key, text, kind, start: kind === 'jsx' ? node.getFullStart() : node.getStart(file), end: node.end });
  };

  function visit(node, display = false, code = false) {
    if (ts.isJsxElement(node)) {
      const tag = node.openingElement.tagName.getText(file);
      visit(node.openingElement, false, code);
      for (const child of node.children) visit(child, true, code || codeElements.has(tag));
      return;
    }
    if (ts.isJsxAttribute(node) || ts.isPropertyAssignment(node)) {
      const fields = ts.isJsxAttribute(node) ? jsxAttributes : displayFields;
      if (node.initializer) visit(node.initializer, fields.has(node.name.getText(file)), code);
      return;
    }
    if (ts.isCallExpression(node)) {
      const name = node.expression.getText(file);
      if (name === 't' || name === 'tr' || name === 'href') return;
      // Function inputs commonly contain identifiers and switch values.
      if (node.expression.kind !== ts.SyntaxKind.ImportKeyword && !display) {
        ts.forEachChild(node, (child) => visit(child, false, code));
        return;
      }
      if (node.expression.kind !== ts.SyntaxKind.ImportKeyword && display) {
        for (const arg of node.arguments) {
          if (ts.isArrowFunction(arg) || ts.isObjectLiteralExpression(arg) || ts.isJsxElement(arg) || ts.isJsxFragment(arg)) visit(arg, false, code);
        }
        return;
      }
    }
    if (ts.isJsxText(node) && !code) add(node, node.text, 'jsx');
    if (ts.isStringLiteralLike(node)) {
      if (!code && displayed(node, display)) {
        const kind = ts.isJsxAttribute(node.parent) ? 'attribute' : 'literal';
        add(node, node.text, kind);
      }
    }
    if (ts.isTemplateExpression(node) && !code && displayed(node, display)) {
      const spans = [node.head, ...node.templateSpans.map((span) => span.literal)];
      const key = compact(spans.map((span) => span.text).join(' ◇ '));
      if (isProse(key)) messages.push({
        key,
        kind: 'template',
        text: spans.map((span) => span.text).join('◇'),
        start: node.getStart(file),
        end: node.end,
        expressions: node.templateSpans.map((span) => source.slice(span.expression.getStart(file), span.expression.end)),
      });
      return;
    }
    ts.forEachChild(node, (child) => visit(child, display, code));
  }
  visit(file);
  return messages;
}
