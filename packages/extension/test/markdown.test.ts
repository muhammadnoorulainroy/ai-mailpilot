/**
 * Tests for the markdown renderer used in chat bubbles, covering inline
 * formatting, lists, headings, code blocks, safe link handling, and HTML
 * escaping that guards against XSS from model output.
 */
import { describe, it, expect } from 'vitest';
import { renderMarkdown, escapeHtml } from '../src/ui/shared/markdown.js';

describe('renderMarkdown', () => {
  it('renders bold, italic, inline code, and a numbered list', () => {
    const html = renderMarkdown('Your score was **17/20** and the *average* was `14.32`.');
    expect(html).toContain('<strong>17/20</strong>');
    expect(html).toContain('<em>average</em>');
    expect(html).toContain('<code>14.32</code>');

    const list = renderMarkdown('1. First\n2. Second');
    expect(list).toContain('<ol>');
    expect(list).toContain('<li>First</li>');
    expect(list).toContain('<li>Second</li>');
  });

  it('renders bullet lists and headings', () => {
    const html = renderMarkdown('## Results\n- a\n- b');
    expect(html).toContain('<h4>Results</h4>');
    expect(html).toContain('<ul>');
    expect(html).toMatch(/<li>a<\/li>\s*<li>b<\/li>/);
  });

  it('keeps blank-line-separated bullets in a single list (no extra spacing)', () => {
    const html = renderMarkdown('- a\n\n- b\n\n- c');
    expect(html.match(/<ul>/g) ?? []).toHaveLength(1);
    expect(html.match(/<\/ul>/g) ?? []).toHaveLength(1);
    expect(html).not.toContain('<p></p>');
  });

  it('puts NO newline between block elements (chat bubble is white-space: pre-wrap)', () => {
    expect(renderMarkdown('First para\n\nSecond para\n\n- item')).toBe(
      '<p>First para</p><p>Second para</p><ul><li>item</li></ul>',
    );
  });

  it('preserves newlines INSIDE a code block', () => {
    expect(renderMarkdown('```\nline1\nline2\n```')).toBe('<pre><code>line1\nline2\n</code></pre>');
  });

  it('renders safe links and rejects javascript: schemes', () => {
    expect(renderMarkdown('see [the site](https://ameli.fr)')).toContain(
      '<a href="https://ameli.fr" target="_blank" rel="noopener noreferrer">the site</a>',
    );
    const evil = renderMarkdown('[click](javascript:alert(1))');
    expect(evil).not.toContain('<a ');
    expect(evil).not.toContain('javascript:alert(1)</a>');
  });

  it('escapes HTML so model output cannot inject markup (XSS)', () => {
    const html = renderMarkdown('<img src=x onerror=alert(1)> and <script>alert(2)</script>');
    expect(html).not.toContain('<img');
    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;img');
    expect(html).toContain('&lt;script&gt;');
  });

  it('escapeHtml escapes the dangerous characters', () => {
    expect(escapeHtml(`<a href="x">'&'</a>`)).toBe('&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;');
  });
});
