/**
 * Minimal markdown-to-HTML renderer for the extension UI, with HTML escaping
 * applied before any pattern runs so the output is safe for innerHTML.
 */

const HTML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

/**
 * Escape the HTML-significant characters in a string so it can be safely embedded in markup.
 */
export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]!);
}

/**
 * Apply inline markdown patterns (code, bold, italic, links) to an
 * already-escaped line, returning HTML.
 */
function renderInline(escaped: string): string {
  return escaped
    .replace(/`([^`]+)`/g, (_m, code: string) => `<code>${code}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, '$1<em>$2</em>')
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g,
      (_m, text: string, url: string) =>
        `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`,
    );
}

/**
 * Render a small subset of markdown to HTML.
 * Safe to assign to innerHTML: input is escaped before any pattern is applied.
 */
export function renderMarkdown(text: string): string {
  const lines = escapeHtml(text).split('\n');
  const out: string[] = [];
  let inCode = false;
  let list: 'ul' | 'ol' | null = null;
  /** Close any open list and reset the tracking state. */
  const closeList = (): void => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      if (inCode) {
        out.push('</code></pre>');
        inCode = false;
      } else {
        closeList();
        out.push('<pre><code>');
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      out.push(`${line}\n`);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(6, heading[1]!.length + 2);
      out.push(`<h${level}>${renderInline(heading[2]!)}</h${level}>`);
      continue;
    }
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ul) {
      if (list !== 'ul') {
        closeList();
        out.push('<ul>');
        list = 'ul';
      }
      out.push(`<li>${renderInline(ul[1]!)}</li>`);
      continue;
    }
    if (ol) {
      if (list !== 'ol') {
        closeList();
        out.push('<ol>');
        list = 'ol';
      }
      out.push(`<li>${renderInline(ol[1]!)}</li>`);
      continue;
    }
    if (line.trim() === '') continue;
    closeList();
    out.push(`<p>${renderInline(line)}</p>`);
  }
  closeList();
  if (inCode) out.push('</code></pre>');
  return out.join('');
}
